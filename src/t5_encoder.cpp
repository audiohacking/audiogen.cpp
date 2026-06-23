#include "t5_encoder.h"

#include <cmath>
#include <cstring>

#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf_util.h"

namespace dasheng {

namespace {

// T5's bidirectional relative position bucket function (encoder self-attn),
// transcribed from transformers' T5Attention._relative_position_bucket.
int32_t relative_position_bucket(int32_t relative_position, int32_t num_buckets, int32_t max_distance) {
    int32_t relative_buckets = 0;
    num_buckets /= 2;
    if (relative_position > 0) {
        relative_buckets += num_buckets;
    }
    int32_t abs_position = relative_position < 0 ? -relative_position : relative_position;

    int32_t max_exact = num_buckets / 2;
    if (abs_position < max_exact) {
        relative_buckets += abs_position;
    } else {
        double scaled = std::log(static_cast<double>(abs_position) / max_exact) /
                         std::log(static_cast<double>(max_distance) / max_exact) * (num_buckets - max_exact);
        int32_t large = max_exact + static_cast<int32_t>(scaled);
        if (large > num_buckets - 1) large = num_buckets - 1;
        relative_buckets += large;
    }
    return relative_buckets;
}

}  // namespace

struct T5Encoder::Impl {
    explicit Impl(const std::string &gguf_path, int n_threads)
        : model(gguf_path), n_threads(n_threads) {
        n_layer = model.kv_i32("t5enc.n_layer");
        n_head = model.kv_i32("t5enc.n_head");
        d_model_ = model.kv_i32("t5enc.d_model");
        d_kv = model.kv_i32("t5enc.d_kv");
        d_ff = model.kv_i32("t5enc.d_ff");
        num_buckets = model.kv_i32("t5enc.relative_attention_num_buckets");
        max_distance = model.kv_i32("t5enc.relative_attention_max_distance");
        eps = model.kv_f32("t5enc.layer_norm_eps");
    }

    GgufModel model;
    int n_threads;
    int n_layer, n_head, d_model_, d_kv, d_ff, num_buckets, max_distance;
    float eps;
};

T5Encoder::T5Encoder(const std::string &gguf_path, int n_threads) : impl_(new Impl(gguf_path, n_threads)) {}

T5Encoder::~T5Encoder() { delete impl_; }

int T5Encoder::d_model() const { return impl_->d_model_; }

std::vector<float> T5Encoder::encode(const std::vector<int32_t> &token_ids) {
    const int64_t seq_len = static_cast<int64_t>(token_ids.size());
    const int n_layer = impl_->n_layer;
    const int n_head = impl_->n_head;
    const int d_model = impl_->d_model_;
    const int d_kv = impl_->d_kv;
    const int d_ff = impl_->d_ff;
    const float eps = impl_->eps;

    // Memory scales with seq_len^2 for attention and seq_len * d_ff for FFN
    const size_t mem_size = 128ull * 1024 * 1024 +
                             static_cast<size_t>(seq_len) * seq_len * n_head * 4 * 16 +
                             static_cast<size_t>(seq_len) * d_ff * 4 * 8 * n_layer;
    struct ggml_init_params cparams = {mem_size, nullptr, false};
    struct ggml_context *ctx = ggml_init(cparams);

    struct ggml_tensor *ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    std::memcpy(ids->data, token_ids.data(), seq_len * sizeof(int32_t));

    struct ggml_tensor *token_embd = impl_->model.tensor("token_embd.weight");
    struct ggml_tensor *cur = ggml_get_rows(ctx, token_embd, ids);  // [d_model, seq_len]

    // Relative position bias, computed once and shared across all layers.
    struct ggml_tensor *rel_b = impl_->model.tensor("enc.blk.0.attn_rel_b.weight");  // [n_head, num_buckets]
    struct ggml_tensor *bucket_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len * seq_len);
    {
        int32_t *dst = static_cast<int32_t *>(bucket_ids->data);
        for (int64_t q = 0; q < seq_len; ++q) {
            for (int64_t k = 0; k < seq_len; ++k) {
                int32_t relative_position = static_cast<int32_t>(k - q);
                dst[q * seq_len + k] =
                    relative_position_bucket(relative_position, impl_->num_buckets, impl_->max_distance);
            }
        }
    }
    struct ggml_tensor *bias = ggml_get_rows(ctx, rel_b, bucket_ids);  // [n_head, seq_k*seq_q]
    bias = ggml_reshape_3d(ctx, bias, n_head, seq_len, seq_len);       // [n_head, seq_k, seq_q]
    bias = ggml_cont(ctx, ggml_permute(ctx, bias, 2, 0, 1, 3));        // [seq_k, seq_q, n_head, 1]

    for (int il = 0; il < n_layer; ++il) {
        const std::string p = "enc.blk." + std::to_string(il) + ".";

        struct ggml_tensor *rms = ggml_rms_norm(ctx, cur, eps);
        struct ggml_tensor *normed = ggml_mul(ctx, rms, impl_->model.tensor(p + "attn_norm.weight"));

        struct ggml_tensor *q = ggml_mul_mat(ctx, impl_->model.tensor(p + "attn_q.weight"), normed);
        struct ggml_tensor *k = ggml_mul_mat(ctx, impl_->model.tensor(p + "attn_k.weight"), normed);
        struct ggml_tensor *v = ggml_mul_mat(ctx, impl_->model.tensor(p + "attn_v.weight"), normed);

        q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, q, d_kv, n_head, seq_len), 0, 2, 1, 3));  // [d_kv, seq_q, n_head]
        k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, k, d_kv, n_head, seq_len), 0, 2, 1, 3));  // [d_kv, seq_k, n_head]
        v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, d_kv, n_head, seq_len), 1, 2, 0, 3));  // [seq_k, d_kv, n_head]

        struct ggml_tensor *scores = ggml_mul_mat(ctx, k, q);  // [seq_k, seq_q, n_head]
        scores = ggml_add(ctx, scores, bias);  // Add relative position bias
        struct ggml_tensor *attn = ggml_soft_max(ctx, scores);

        struct ggml_tensor *kqv = ggml_mul_mat(ctx, v, attn);  // [d_kv, seq_q, n_head]
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));  // [d_kv, n_head, seq_q]
        kqv = ggml_reshape_2d(ctx, kqv, d_model, seq_len);

        struct ggml_tensor *attn_out = ggml_mul_mat(ctx, impl_->model.tensor(p + "attn_o.weight"), kqv);
        cur = ggml_add(ctx, cur, attn_out);

        struct ggml_tensor *normed2 = ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), impl_->model.tensor(p + "ffn_norm.weight"));
        struct ggml_tensor *gate = ggml_gelu(ctx, ggml_mul_mat(ctx, impl_->model.tensor(p + "ffn_gate.weight"), normed2));
        struct ggml_tensor *up = ggml_mul_mat(ctx, impl_->model.tensor(p + "ffn_up.weight"), normed2);
        struct ggml_tensor *ff = ggml_mul(ctx, gate, up);
        struct ggml_tensor *down = ggml_mul_mat(ctx, impl_->model.tensor(p + "ffn_down.weight"), ff);
        cur = ggml_add(ctx, cur, down);
    }

    cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), impl_->model.tensor("enc.output_norm.weight"));

    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, cur);
    ggml_graph_compute_with_ctx(ctx, graph, impl_->n_threads);

    std::vector<float> result(static_cast<size_t>(seq_len) * d_model);
    std::memcpy(result.data(), cur->data, result.size() * sizeof(float));

    ggml_free(ctx);
    return result;
}

}  // namespace dasheng
