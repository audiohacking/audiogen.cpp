#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Minimal T5 encoder-only graph (flan-t5-large), hand-written against ggml --
// no KV-cache, no decoder, no sampling. Only the forward pass needed to
// produce per-token hidden states for the DiT's cross-attention context.
// Weights come from convert/convert_t5_encoder.py's GGUF (see that script's
// docstring for the tensor naming scheme).
namespace dasheng {

class T5Encoder {
public:
    explicit T5Encoder(const std::string &gguf_path, int n_threads = 4);
    ~T5Encoder();

    int d_model() const;

    // Returns hidden states of shape [seq_len, d_model], row-major.
    std::vector<float> encode(const std::vector<int32_t> &token_ids);

private:
    struct Impl;
    Impl *impl_;
};

}  // namespace dasheng
