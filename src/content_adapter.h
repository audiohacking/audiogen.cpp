#pragma once

#include <vector>

#include "gguf_util.h"

// content_encoder.text_encoder.proj + CrossAttentionAdapter, restricted to the
// text-to-audio path (is_time_aligned == false for every item): the
// DurationPredictor's per-token output and the FastSpeech-style alignment
// path it would otherwise feed are dead code on this path -- the model
// unconditionally overwrites time_aligned_content with dummy_ta_embed
// whenever is_time_aligned is false (see _get_backbone_input), and only the
// global_duration_mlp branch survives into _prepare_global_duration. See
// reference/modeling_dasheng_audiogen.py for the full (general) path.
namespace dasheng {

struct ContentAdapterOutput {
    std::vector<float> context;  // [context_len, content_dim] row-major
    int context_len = 0;
    int global_latent_length = 0;
    std::vector<float> time_aligned_content;  // [global_latent_length, content_dim] row-major
};

class ContentAdapter {
public:
    explicit ContentAdapter(GgufModel &dit_model, int n_threads = 4);
    ~ContentAdapter();

    // t5_hidden: T5 encoder output, [seq_len, content_dim] row-major (pre
    // content_encoder.text_encoder.proj).
    ContentAdapterOutput run(const std::vector<float> &t5_hidden, int seq_len);

private:
    struct Impl;
    Impl *impl_;
};

}  // namespace dasheng
