#include "tokenizer.h"

#include <stdexcept>

#include <sentencepiece_processor.h>

namespace dasheng {

struct T5Tokenizer::Impl {
    sentencepiece::SentencePieceProcessor sp;
};

T5Tokenizer::T5Tokenizer(const std::string &spiece_model_path) : impl_(new Impl()) {
    const auto status = impl_->sp.Load(spiece_model_path);
    if (!status.ok()) {
        delete impl_;
        throw std::runtime_error("failed to load sentencepiece model: " + spiece_model_path +
                                  " (" + status.ToString() + ")");
    }
}

T5Tokenizer::~T5Tokenizer() { delete impl_; }

std::vector<int32_t> T5Tokenizer::encode(const std::string &text) const {
    std::vector<int> ids;
    const auto status = impl_->sp.Encode(text, &ids);
    if (!status.ok()) {
        throw std::runtime_error("sentencepiece encode failed: " + status.ToString());
    }
    std::vector<int32_t> result(ids.begin(), ids.end());
    result.push_back(1);  // </s> EOS, matching transformers.T5Tokenizer
    return result;
}

}  // namespace dasheng
