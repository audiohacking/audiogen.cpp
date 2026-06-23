#pragma once

#include <string>
#include <vector>

#include "gguf_util.h"

// LayerFusionAudioDiT backbone: 16 in_blocks + 1 mid_block + 16 out_blocks,
// embed_dim 1536, 24 heads, head_dim 64. time_fusion='ada' (per-block AdaLN),
// context_fusion='cross' (T5 content via cross-attention),
// ta_context_fusion='add' (per-layer "time-aligned content" added after
// self-attn, recomputed every layer from the same constant
// time_aligned_content input), rope_mode='shared' (RoPE on self-attention
// only), qk_norm='layernorm' (applied before RoPE), pe_method='none' (no
// absolute position embeddings anywhere). See reference/dit.py for the full
// trace this implementation follows.
namespace dasheng {

class DiT {
public:
    explicit DiT(const std::string &gguf_path, int n_threads = 4);
    ~DiT();

    int latent_dim() const;  // 1280
    int embed_dim() const;   // 1536

    // x: [T, latent_dim] row-major -- the noisy latent.
    // timestep: scalar diffusion timestep, same convention as the reference
    // TimestepEmbedder (raw timestep value, not normalized to [0,1]).
    // context: [context_len, content_dim] row-major
    // (ContentAdapterOutput::context).
    // time_aligned_content: [T, content_dim] row-major
    // (ContentAdapterOutput::time_aligned_content; length must equal T).
    // Returns the predicted velocity field, [T, latent_dim] row-major.
    std::vector<float> forward(const std::vector<float> &x, int T, float timestep,
                                const std::vector<float> &context, int context_len,
                                const std::vector<float> &time_aligned_content);

private:
    struct Impl;
    Impl *impl_;
};

}  // namespace dasheng
