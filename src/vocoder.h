#pragma once

#include <string>
#include <vector>

#include "gguf_util.h"

// ConvTranspose1d "upsampler" (kernel=stride=2) + Vocos decoder: Conv1d embed
// (k=7,p=3) -> LayerNorm -> 12x ConvNeXtBlock -> final LayerNorm -> ISTFTHead
// (Linear -> magnitude/phase -> custom "same"-padding ISTFT via pocketfft).
// See reference/vocos.py and reference/modeling_dasheng_tokenizer.py.
namespace dasheng {

class Vocoder {
public:
    explicit Vocoder(const std::string &gguf_path, int n_threads = 4);
    ~Vocoder();

    // latent: [T, latent_dim] row-major (the DiT-denoised latent).
    // Returns the reconstructed waveform, length == T * 2 * istft_hop
    // (2x from the ConvTranspose1d upsampler, istft_hop from the ISTFT head).
    std::vector<float> decode(const std::vector<float> &latent, int T);

private:
    struct Impl;
    Impl *impl_;
};

}  // namespace dasheng
