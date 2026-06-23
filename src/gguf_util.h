#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "gguf.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "backend.h"

// GGUF model loading utilities for CPU and GPU backends.
// - GgufModel: CPU-only loading (original behavior)
// - GgufModelGPU: Loads weights directly to GPU using WeightCtx pattern
namespace dasheng {

// CPU-only GGUF loader (original behavior, kept for compatibility)
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

// GPU-aware GGUF loader using mmap + WeightCtx for direct GPU transfer.
// Following acestep.cpp pattern for efficient GPU weight loading.
class GgufModelGPU {
public:
    explicit GgufModelGPU(const std::string &path, ggml_backend_t backend)
        : path_(path), backend_(backend) {

        // Step 1: Load GGUF with no_alloc=true (metadata only)
        struct gguf_init_params params = {/*.no_alloc =*/ true, /*.ctx =*/ &ctx_};
        gguf_ = gguf_init_from_file(path.c_str(), params);
        if (!gguf_) {
            throw std::runtime_error("failed to load GGUF file: " + path);
        }

        // Step 2: Memory-map the file for direct data access
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("failed to open GGUF file: " + path);
        }

        struct stat st;
        if (fstat(fd_, &st) < 0) {
            close(fd_);
            throw std::runtime_error("failed to stat GGUF file: " + path);
        }
        file_size_ = static_cast<size_t>(st.st_size);

        mmap_data_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mmap_data_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("failed to mmap GGUF file: " + path);
        }

        // Get data offset (where tensor data starts in file)
        data_offset_ = gguf_get_data_offset(gguf_);

        // Step 3: Allocate all tensors on the backend
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
        if (!buffer_) {
            munmap(mmap_data_, file_size_);
            close(fd_);
            throw std::runtime_error("failed to allocate tensors on backend");
        }
        ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        // Step 4: Copy tensor data from mmap to backend
        size_t total_bytes = 0;
        int64_t n_tensors = gguf_get_n_tensors(gguf_);
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char* name = gguf_get_tensor_name(gguf_, i);
            size_t offset = gguf_get_tensor_offset(gguf_, i);
            size_t size = gguf_get_tensor_size(gguf_, i);

            struct ggml_tensor* t = ggml_get_tensor(ctx_, name);
            if (t) {
                const void* src = static_cast<const char*>(mmap_data_) + data_offset_ + offset;
                ggml_backend_tensor_set(t, src, 0, size);
                total_bytes += size;
            }
        }

        std::fprintf(stderr, "[%s] loaded %.1f MB to %s\n",
                     path.c_str(), total_bytes / (1024.0 * 1024.0),
                     ggml_backend_name(backend));
    }

    ~GgufModelGPU() {
        if (buffer_) ggml_backend_buffer_free(buffer_);
        if (ctx_) ggml_free(ctx_);
        if (gguf_) gguf_free(gguf_);
        if (mmap_data_ && mmap_data_ != MAP_FAILED) {
            munmap(mmap_data_, file_size_);
        }
        if (fd_ >= 0) close(fd_);
    }

    GgufModelGPU(const GgufModelGPU &) = delete;
    GgufModelGPU &operator=(const GgufModelGPU &) = delete;

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

    ggml_backend_t backend() const { return backend_; }

private:
    int64_t find_key(const std::string &key) const {
        int64_t id = gguf_find_key(gguf_, key.c_str());
        if (id < 0) {
            throw std::runtime_error("GGUF file is missing metadata key: " + key);
        }
        return id;
    }

    std::string path_;
    ggml_backend_t backend_;
    struct gguf_context *gguf_ = nullptr;
    struct ggml_context *ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;

    // mmap state
    int fd_ = -1;
    void* mmap_data_ = nullptr;
    size_t file_size_ = 0;
    size_t data_offset_ = 0;
};

}  // namespace dasheng
