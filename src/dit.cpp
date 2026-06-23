#include "dit.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"

namespace dasheng {

namespace {

constexpr int kEmbedDim = 1536;
constexpr int kLatentDim = 1280;
constexpr int kContentDim = 1024;
constexpr int kNumHeads = 24;
constexpr int kHeadDim = kEmbedDim / kNumHeads;  // 64
constexpr int kFreqEmbedDim = 256;
constexpr int kNumHalfBlocks = 16;  // in_blocks == out_blocks count
constexpr float kLnEps = 1e-5f;

// timestep_embedding() from reference/modules.py: half=dim/2, freqs[i] =
// exp(-log(max_period) * i / half), embedding = cat(cos(t*freqs), sin(t*freqs)).
std::vector<float> sinusoidal_timestep_embedding(float t, int dim, float max_period = 10000.0f) {
    std::vector<float> out(dim);
    const int half = dim / 2;
    for (int i = 0; i < half; ++i) {
        const float freq = std::exp(-std::log(max_period) * static_cast<float>(i) / static_cast<float>(half));
        const float arg = t * freq;
        out[i] = std::cos(arg);
        out[half + i] = std::sin(arg);
    }
    return out;
}

}  // namespace

struct DiT::Impl {
    explicit Impl(const std::string &gguf_path, int n_threads) : model(gguf_path), n_threads(n_threads) {}
    GgufModel model;
    int n_threads;
};

DiT::DiT(const std::string &gguf_path, int n_threads) : impl_(new Impl(gguf_path, n_threads)) {}

DiT::~DiT() { delete impl_; }

int DiT::latent_dim() const { return kLatentDim; }
int DiT::embed_dim() const { return kEmbedDim; }

std::vector<float> DiT::forward(const std::vector<float> &x, int T, float timestep,
                                 const std::vector<float> &context, int context_len,
                                 const std::vector<float> &time_aligned_content) {
    GgufModel &model = impl_->model;

    const size_t per_block_bytes = static_cast<size_t>(T) * T * 400 +
                                    static_cast<size_t>(T) * context_len * 200 +
                                    static_cast<size_t>(T) * 400000;
    const size_t mem_size = 512ull * 1024 * 1024 + per_block_bytes * 34 * 3;

    struct ggml_init_params cparams = {mem_size, nullptr, false};
    struct ggml_context *ctx = ggml_init(cparams);
    if (!ctx) {
        throw std::runtime_error("DiT::forward: ggml_init failed (mem_size too large?)");
    }

    // --- inputs ---
    struct ggml_tensor *x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kLatentDim, T);
    std::memcpy(x_in->data, x.data(), x.size() * sizeof(float));

    struct ggml_tensor *context_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kContentDim, context_len);
    std::memcpy(context_in->data, context.data(), context.size() * sizeof(float));

    struct ggml_tensor *ta_content_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kContentDim, T);
    std::memcpy(ta_content_in->data, time_aligned_content.data(), time_aligned_content.size() * sizeof(float));

    struct ggml_tensor *positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    for (int i = 0; i < T; ++i) {
        static_cast<int32_t *>(positions->data)[i] = i;
    }

    auto linear = [&](struct ggml_tensor *w, struct ggml_tensor *b, struct ggml_tensor *in) {
        struct ggml_tensor *out = ggml_mul_mat(ctx, w, in);
        if (b) out = ggml_add(ctx, out, b);
        return out;
    };

    auto layer_norm = [&](struct ggml_tensor *in, const std::string &prefix) {
        struct ggml_tensor *normed = ggml_norm(ctx, in, kLnEps);
        normed = ggml_mul(ctx, normed, model.tensor(prefix + ".weight"));
        normed = ggml_add(ctx, normed, model.tensor(prefix + ".bias"));
        return normed;
    };

    // film_modulate(x) = x*(1+scale)+shift = x*scale + x + shift
    auto film = [&](struct ggml_tensor *normed, struct ggml_tensor *shift, struct ggml_tensor *scale) {
        struct ggml_tensor *out = ggml_add(ctx, ggml_mul(ctx, normed, scale), normed);
        out = ggml_add(ctx, out, shift);
        return out;
    };

    // --- patch_embed: Conv1d(1280,1536,k=1,s=1) === per-token Linear ---
    struct ggml_tensor *patch_w =
        ggml_reshape_2d(ctx, model.tensor("backbone.patch_embed.proj.weight"), kLatentDim, kEmbedDim);
    struct ggml_tensor *x_h = linear(patch_w, model.tensor("backbone.patch_embed.proj.bias"), x_in);  // [1536,T]

    // --- context_embed: Linear(1024->1536) -> SiLU -> Linear(1536->1536) ---
    struct ggml_tensor *ctx_h =
        linear(model.tensor("backbone.context_embed.0.weight"), model.tensor("backbone.context_embed.0.bias"), context_in);
    ctx_h = ggml_silu(ctx, ctx_h);
    ctx_h = linear(model.tensor("backbone.context_embed.2.weight"), model.tensor("backbone.context_embed.2.bias"), ctx_h);  // [1536, context_len]

    // --- time embedding ---
    std::vector<float> t_freq = sinusoidal_timestep_embedding(timestep, kFreqEmbedDim);
    struct ggml_tensor *t_freq_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kFreqEmbedDim);
    std::memcpy(t_freq_t->data, t_freq.data(), t_freq.size() * sizeof(float));

    struct ggml_tensor *time_token =
        linear(model.tensor("backbone.time_embed.mlp.0.weight"), model.tensor("backbone.time_embed.mlp.0.bias"), t_freq_t);
    time_token = ggml_silu(ctx, time_token);
    time_token =
        linear(model.tensor("backbone.time_embed.mlp.2.weight"), model.tensor("backbone.time_embed.mlp.2.bias"), time_token);
    time_token = ggml_silu(ctx, time_token);  // top-level self.time_act, [1536]

    struct ggml_tensor *time_ada_final =
        linear(model.tensor("backbone.time_ada_final.weight"), model.tensor("backbone.time_ada_final.bias"), time_token);  // [3072]
    struct ggml_tensor *shift_final = ggml_view_1d(ctx, time_ada_final, kEmbedDim, 0);
    struct ggml_tensor *scale_final = ggml_view_1d(ctx, time_ada_final, kEmbedDim, static_cast<size_t>(kEmbedDim) * time_ada_final->nb[0]);

    auto qk_layernorm = [&](struct ggml_tensor *t, const std::string &prefix) {
        struct ggml_tensor *normed = ggml_norm(ctx, t, kLnEps);
        normed = ggml_mul(ctx, normed, model.tensor(prefix + ".weight"));
        normed = ggml_add(ctx, normed, model.tensor(prefix + ".bias"));
        return normed;
    };

    auto self_attention = [&](struct ggml_tensor *x_norm, const std::string &prefix) {
        struct ggml_tensor *q = ggml_mul_mat(ctx, model.tensor(prefix + ".to_q.weight"), x_norm);
        struct ggml_tensor *k = ggml_mul_mat(ctx, model.tensor(prefix + ".to_k.weight"), x_norm);
        struct ggml_tensor *v = ggml_mul_mat(ctx, model.tensor(prefix + ".to_v.weight"), x_norm);

        q = ggml_reshape_3d(ctx, q, kHeadDim, kNumHeads, T);
        k = ggml_reshape_3d(ctx, k, kHeadDim, kNumHeads, T);
        v = ggml_reshape_3d(ctx, v, kHeadDim, kNumHeads, T);

        q = qk_layernorm(q, prefix + ".norm_q");
        k = qk_layernorm(k, prefix + ".norm_k");

        q = ggml_rope(ctx, q, positions, kHeadDim, GGML_ROPE_TYPE_NEOX);
        k = ggml_rope(ctx, k, positions, kHeadDim, GGML_ROPE_TYPE_NEOX);

        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [head_dim, T, n_head]
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [head_dim, T, n_head]
        v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [T, head_dim, n_head]

        struct ggml_tensor *scores = ggml_mul_mat(ctx, k, q);  // [T, T, n_head]
        struct ggml_tensor *attn = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f / std::sqrt(static_cast<float>(kHeadDim)), 0.0f);

        struct ggml_tensor *kqv = ggml_mul_mat(ctx, v, attn);             // [head_dim, T, n_head]
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));        // [head_dim, n_head, T]
        kqv = ggml_reshape_2d(ctx, kqv, kEmbedDim, T);

        return linear(model.tensor(prefix + ".proj.weight"), model.tensor(prefix + ".proj.bias"), kqv);
    };

    auto cross_attention = [&](struct ggml_tensor *x_norm, struct ggml_tensor *context_normed, const std::string &prefix) {
        struct ggml_tensor *q = ggml_mul_mat(ctx, model.tensor(prefix + ".to_q.weight"), x_norm);
        struct ggml_tensor *k = ggml_mul_mat(ctx, model.tensor(prefix + ".to_k.weight"), context_normed);
        struct ggml_tensor *v = ggml_mul_mat(ctx, model.tensor(prefix + ".to_v.weight"), context_normed);

        q = ggml_reshape_3d(ctx, q, kHeadDim, kNumHeads, T);
        k = ggml_reshape_3d(ctx, k, kHeadDim, kNumHeads, context_len);
        v = ggml_reshape_3d(ctx, v, kHeadDim, kNumHeads, context_len);

        q = qk_layernorm(q, prefix + ".norm_q");
        k = qk_layernorm(k, prefix + ".norm_k");

        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [head_dim, T, n_head]
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [head_dim, context_len, n_head]
        v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [context_len, head_dim, n_head]

        struct ggml_tensor *scores = ggml_mul_mat(ctx, k, q);  // [context_len, T, n_head]
        struct ggml_tensor *attn = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f / std::sqrt(static_cast<float>(kHeadDim)), 0.0f);

        struct ggml_tensor *kqv = ggml_mul_mat(ctx, v, attn);             // [head_dim, T, n_head]
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));        // [head_dim, n_head, T]
        kqv = ggml_reshape_2d(ctx, kqv, kEmbedDim, T);

        return linear(model.tensor(prefix + ".proj.weight"), model.tensor(prefix + ".proj.bias"), kqv);
    };

    auto geglu_mlp = [&](struct ggml_tensor *x_norm, const std::string &prefix) {
        struct ggml_tensor *proj =
            linear(model.tensor(prefix + ".net.0.proj.weight"), model.tensor(prefix + ".net.0.proj.bias"), x_norm);  // [12288,T]
        const int64_t inner = proj->ne[0] / 2;  // 6144
        struct ggml_tensor *hidden = ggml_cont(ctx, ggml_view_2d(ctx, proj, inner, proj->ne[1], proj->nb[1], 0));
        struct ggml_tensor *gate =
            ggml_cont(ctx, ggml_view_2d(ctx, proj, inner, proj->ne[1], proj->nb[1], static_cast<size_t>(inner) * proj->nb[0]));
        struct ggml_tensor *act = ggml_mul(ctx, hidden, ggml_gelu_erf(ctx, gate));
        return linear(model.tensor(prefix + ".net.2.weight"), model.tensor(prefix + ".net.2.bias"), act);
    };

    auto run_block = [&](struct ggml_tensor *x, const std::string &prefix, struct ggml_tensor *skip) {
        if (skip) {
            struct ggml_tensor *cat = ggml_concat(ctx, x, skip, 0);  // [3072,T]
            cat = layer_norm(cat, prefix + ".skip_norm");
            x = linear(model.tensor(prefix + ".skip_linear.weight"), model.tensor(prefix + ".skip_linear.bias"), cat);
        }

        struct ggml_tensor *time_ada =
            linear(model.tensor(prefix + ".adaln.time_ada.weight"), model.tensor(prefix + ".adaln.time_ada.bias"), time_token);  // [9216]
        auto chunk = [&](int i) { return ggml_view_1d(ctx, time_ada, kEmbedDim, static_cast<size_t>(i) * kEmbedDim * time_ada->nb[0]); };
        struct ggml_tensor *shift_msa = chunk(0);
        struct ggml_tensor *scale_msa = chunk(1);
        struct ggml_tensor *gate_msa = chunk(2);
        struct ggml_tensor *shift_mlp = chunk(3);
        struct ggml_tensor *scale_mlp = chunk(4);
        struct ggml_tensor *gate_mlp = chunk(5);

        // self-attention (RoPE, qk-layernorm), tanh(1-gate) gating
        struct ggml_tensor *x_norm1 = film(layer_norm(x, prefix + ".norm1"), shift_msa, scale_msa);
        struct ggml_tensor *attn_out = self_attention(x_norm1, prefix + ".attn");
        struct ggml_tensor *tanh_gate_msa = ggml_tanh(ctx, ggml_scale_bias(ctx, gate_msa, -1.0f, 1.0f));
        x = ggml_add(ctx, x, ggml_mul(ctx, attn_out, tanh_gate_msa));

        // ta_context "add" fusion: recomputed every layer from the constant
        // (non-evolving) time_aligned_content input.
        struct ggml_tensor *ta_normed = layer_norm(ta_content_in, prefix + ".ta_context_norm");
        struct ggml_tensor *ta_proj = ggml_mul_mat(ctx, model.tensor(prefix + ".ta_context_projection.weight"), ta_normed);
        x = ggml_add(ctx, x, ta_proj);

        // cross-attention to T5 content, no gating
        struct ggml_tensor *x_norm2 = layer_norm(x, prefix + ".norm2");
        struct ggml_tensor *context_normed = layer_norm(ctx_h, prefix + ".norm_context");
        struct ggml_tensor *cross_out = cross_attention(x_norm2, context_normed, prefix + ".cross_attn");
        x = ggml_add(ctx, x, cross_out);

        // GEGLU FFN, (1-gate) plain gating
        struct ggml_tensor *x_norm3 = film(layer_norm(x, prefix + ".norm3"), shift_mlp, scale_mlp);
        struct ggml_tensor *mlp_out = geglu_mlp(x_norm3, prefix + ".mlp");
        struct ggml_tensor *one_minus_gate_mlp = ggml_scale_bias(ctx, gate_mlp, -1.0f, 1.0f);
        x = ggml_add(ctx, x, ggml_mul(ctx, mlp_out, one_minus_gate_mlp));

        return x;
    };

    std::vector<struct ggml_tensor *> skips;
    skips.reserve(kNumHalfBlocks);

    for (int i = 0; i < kNumHalfBlocks; ++i) {
        x_h = run_block(x_h, "backbone.in_blocks." + std::to_string(i), nullptr);
        skips.push_back(x_h);
    }

    x_h = run_block(x_h, "backbone.mid_block", nullptr);

    for (int i = 0; i < kNumHalfBlocks; ++i) {
        struct ggml_tensor *skip = skips.back();
        skips.pop_back();
        x_h = run_block(x_h, "backbone.out_blocks." + std::to_string(i), skip);
    }

    // --- final_block ---
    struct ggml_tensor *final_normed = film(layer_norm(x_h, "backbone.final_block.norm"), shift_final, scale_final);
    struct ggml_tensor *final_lin =
        linear(model.tensor("backbone.final_block.linear.weight"), model.tensor("backbone.final_block.linear.bias"), final_normed);  // [1280,T]

    // unpatchify is a no-op (patch_size=1); final_layer is a real Conv1d(1280,1280,k=3,p=1).
    struct ggml_tensor *conv_in = ggml_cont(ctx, ggml_transpose(ctx, final_lin));  // [T,1280]
    conv_in = ggml_reshape_3d(ctx, conv_in, T, kLatentDim, 1);
    struct ggml_tensor *conv_out =
        ggml_conv_1d(ctx, model.tensor("backbone.final_block.final_layer.weight"), conv_in, 1, 1, 1);  // [T,1280,1]
    conv_out = ggml_reshape_2d(ctx, conv_out, T, kLatentDim);
    conv_out = ggml_cont(ctx, ggml_transpose(ctx, conv_out));  // [1280,T]
    conv_out = ggml_add(ctx, conv_out, model.tensor("backbone.final_block.final_layer.bias"));

    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 1u << 16, false);
    ggml_build_forward_expand(graph, conv_out);
    ggml_graph_compute_with_ctx(ctx, graph, impl_->n_threads);

    std::vector<float> result(static_cast<size_t>(T) * kLatentDim);
    std::memcpy(result.data(), conv_out->data, result.size() * sizeof(float));

    ggml_free(ctx);
    return result;
}

}  // namespace dasheng
