#pragma once

#include <stdexcept>
#include <string>

#include "gguf.h"
#include "ggml.h"

// Thin RAII wrapper for loading a GGUF file's tensors (with data already
// resident, no_alloc=false) plus typed metadata accessors. Shared by
// T5Encoder, DiT and Vocoder -- each owns one of these for its weights.
namespace dasheng {

class GgufModel {
public:
    explicit GgufModel(const std::string &path) {
        struct gguf_init_params params = {/*.no_alloc =*/ false, /*.ctx =*/ &ctx_};
        gguf_ = gguf_init_from_file(path.c_str(), params);
        if (!gguf_) {
            throw std::runtime_error("failed to load GGUF file: " + path);
        }
    }

    ~GgufModel() {
        if (ctx_) ggml_free(ctx_);
        if (gguf_) gguf_free(gguf_);
    }

    GgufModel(const GgufModel &) = delete;
    GgufModel &operator=(const GgufModel &) = delete;

    struct ggml_context *ctx() const { return ctx_; }

    struct ggml_tensor *tensor(const std::string &name) const {
        struct ggml_tensor *t = ggml_get_tensor(ctx_, name.c_str());
        if (!t) {
            throw std::runtime_error("GGUF file is missing tensor: " + name);
        }
        return t;
    }

    bool has_tensor(const std::string &name) const {
        return ggml_get_tensor(ctx_, name.c_str()) != nullptr;
    }

    int32_t kv_i32(const std::string &key) const {
        return gguf_get_val_i32(gguf_, find_key(key));
    }

    float kv_f32(const std::string &key) const {
        return gguf_get_val_f32(gguf_, find_key(key));
    }

    std::string kv_str(const std::string &key) const {
        return gguf_get_val_str(gguf_, find_key(key));
    }

    bool kv_bool(const std::string &key) const {
        return gguf_get_val_bool(gguf_, find_key(key));
    }

private:
    int64_t find_key(const std::string &key) const {
        int64_t id = gguf_find_key(gguf_, key.c_str());
        if (id < 0) {
            throw std::runtime_error("GGUF file is missing metadata key: " + key);
        }
        return id;
    }

    struct gguf_context *gguf_ = nullptr;
    struct ggml_context *ctx_ = nullptr;
};

}  // namespace dasheng
