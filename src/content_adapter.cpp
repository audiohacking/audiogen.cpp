#include "content_adapter.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "ggml-cpu.h"
#include "ggml.h"

namespace dasheng {

namespace {

constexpr int kContentDim = 1024;
constexpr int kNumHeads = 16;
constexpr int kHeadDim = kContentDim / kNumHeads;
constexpr int kInstructionLen = 14;
constexpr float kDurationOffset = 1.0f;
constexpr int kLatentTokenRate = 25;  // sample_rate(16000) / downsampling_ratio(640)

std::vector<float> tensor_to_f32(const struct ggml_tensor *t) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t *>(t->data), out.data(), n);
    } else {
        throw std::runtime_error("content_adapter: unsupported tensor dtype");
    }
    return out;
}

}  // namespace

struct ContentAdapter::Impl {
    explicit Impl(GgufModel &m, int n_threads) : model(m), n_threads(n_threads) {}
    GgufModel &model;
    int n_threads;
};

ContentAdapter::ContentAdapter(GgufModel &dit_model, int n_threads) : impl_(new Impl(dit_model, n_threads)) {}

ContentAdapter::~ContentAdapter() { delete impl_; }

ContentAdapterOutput ContentAdapter::run(const std::vector<float> &t5_hidden, int seq_len) {
    GgufModel &model = impl_->model;

    const size_t mem_size = 32ull * 1024 * 1024;
    struct ggml_init_params cparams = {mem_size, nullptr, false};
    struct ggml_context *ctx = ggml_init(cparams);

    struct ggml_tensor *h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kContentDim, seq_len);
    std::memcpy(h->data, t5_hidden.data(), t5_hidden.size() * sizeof(float));

    // content_encoder.text_encoder.proj
    struct ggml_tensor *content = ggml_add(
        ctx, ggml_mul_mat(ctx, model.tensor("content_encoder.text_encoder.proj.weight"), h),
        model.tensor("content_encoder.text_encoder.proj.bias"));

    struct ggml_tensor *instruction =
        ggml_reshape_2d(ctx, model.tensor("instruction_embedding"), kContentDim, kInstructionLen);

    struct ggml_tensor *in_proj_w = model.tensor("content_adapter.attn.in_proj_weight");  // [1024, 3072]
    struct ggml_tensor *in_proj_b = model.tensor("content_adapter.attn.in_proj_bias");    // [3072]

    auto slice_weight_chunk = [&](int chunk) {
        return ggml_cont(ctx, ggml_view_2d(ctx, in_proj_w, kContentDim, kContentDim, in_proj_w->nb[1],
                                            static_cast<size_t>(chunk) * kContentDim * in_proj_w->nb[1]));
    };
    auto slice_bias_chunk = [&](int chunk) {
        return ggml_view_1d(ctx, in_proj_b, kContentDim, static_cast<size_t>(chunk) * kContentDim * in_proj_b->nb[0]);
    };

    struct ggml_tensor *q = ggml_add(ctx, ggml_mul_mat(ctx, slice_weight_chunk(0), content), slice_bias_chunk(0));
    struct ggml_tensor *k = ggml_add(ctx, ggml_mul_mat(ctx, slice_weight_chunk(1), instruction), slice_bias_chunk(1));
    struct ggml_tensor *v = ggml_add(ctx, ggml_mul_mat(ctx, slice_weight_chunk(2), instruction), slice_bias_chunk(2));

    q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, q, kHeadDim, kNumHeads, seq_len), 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, k, kHeadDim, kNumHeads, kInstructionLen), 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, kHeadDim, kNumHeads, kInstructionLen), 1, 2, 0, 3));

    struct ggml_tensor *scores = ggml_mul_mat(ctx, k, q);  // [14, seq_len, 16]
    struct ggml_tensor *attn = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f / std::sqrt(static_cast<float>(kHeadDim)), 0.0f);

    struct ggml_tensor *kqv = ggml_mul_mat(ctx, v, attn);              // [64, seq_len, 16]
    kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));          // [64, 16, seq_len]
    kqv = ggml_reshape_2d(ctx, kqv, kContentDim, seq_len);

    struct ggml_tensor *attn_out = ggml_add(
        ctx, ggml_mul_mat(ctx, model.tensor("content_adapter.attn.out_proj.weight"), kqv),
        model.tensor("content_adapter.attn.out_proj.bias"));

    struct ggml_tensor *x = ggml_add(ctx, attn_out, content);
    x = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1e-5f), model.tensor("content_adapter.norm.weight")),
                 model.tensor("content_adapter.norm.bias"));

    struct ggml_tensor *cproj_w =
        ggml_reshape_2d(ctx, model.tensor("content_adapter.content_proj.weight"), kContentDim, kContentDim);
    struct ggml_tensor *content2 =
        ggml_add(ctx, ggml_mul_mat(ctx, cproj_w, x), model.tensor("content_adapter.content_proj.bias"));

    struct ggml_tensor *x_t = ggml_cont(ctx, ggml_transpose(ctx, x));  // [seq_len, 1024]
    struct ggml_tensor *pooled = ggml_pool_1d(ctx, x_t, GGML_OP_POOL_AVG, seq_len, seq_len, 0);  // [1, 1024]
    pooled = ggml_reshape_1d(ctx, pooled, kContentDim);

    struct ggml_tensor *gd = ggml_add(ctx, ggml_mul_mat(ctx, model.tensor("content_adapter.global_duration_mlp.0.weight"), pooled),
                                       model.tensor("content_adapter.global_duration_mlp.0.bias"));
    gd = ggml_relu(ctx, gd);
    gd = ggml_add(ctx, ggml_mul_mat(ctx, model.tensor("content_adapter.global_duration_mlp.3.weight"), gd),
                  model.tensor("content_adapter.global_duration_mlp.3.bias"));

    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, content2);
    ggml_build_forward_expand(graph, gd);
    ggml_graph_compute_with_ctx(ctx, graph, impl_->n_threads);

    ContentAdapterOutput result;
    result.context.resize(static_cast<size_t>(seq_len) * kContentDim);
    std::memcpy(result.context.data(), content2->data, result.context.size() * sizeof(float));
    result.context_len = seq_len;

    const float global_duration_pred = static_cast<const float *>(gd->data)[0];
    const float global_duration = std::exp(global_duration_pred) - kDurationOffset;
    result.global_latent_length = static_cast<int>(std::round(global_duration * kLatentTokenRate));

    std::vector<float> dummy_ta = tensor_to_f32(model.tensor("dummy_ta_embed"));
    result.time_aligned_content.resize(static_cast<size_t>(result.global_latent_length) * kContentDim);
    for (int t = 0; t < result.global_latent_length; ++t) {
        std::memcpy(result.time_aligned_content.data() + static_cast<size_t>(t) * kContentDim, dummy_ta.data(),
                    kContentDim * sizeof(float));
    }

    ggml_free(ctx);
    return result;
}

}  // namespace dasheng
