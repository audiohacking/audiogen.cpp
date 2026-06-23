#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dasheng {

// Thin wrapper around google/sentencepiece, loading T5's spiece.model
// directly so the engine has no Python tokenization dependency.
class T5Tokenizer {
public:
    explicit T5Tokenizer(const std::string &spiece_model_path);
    ~T5Tokenizer();

    // Encodes text and appends the EOS token id (1), matching
    // transformers' T5Tokenizer behavior.
    std::vector<int32_t> encode(const std::string &text) const;

private:
    struct Impl;
    Impl *impl_;
};

}  // namespace dasheng
