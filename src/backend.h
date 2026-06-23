#pragma once

#include <memory>
#include <string>

#include "ggml.h"
#include "ggml-backend.h"

namespace dasheng {

// Backend abstraction for CPU/Metal/CUDA compute.
// Manages backend lifecycle and provides graph execution.
class Backend {
public:
    enum class Type {
        CPU,
        Metal,
        CUDA,
        Auto  // Auto-detect best available
    };

    // Initialize backend of specified type.
    // Auto will try Metal -> CUDA -> CPU in order.
    explicit Backend(Type type = Type::Auto);
    ~Backend();

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    // Get the underlying ggml_backend_t
    ggml_backend_t handle() const { return backend_; }

    // Get the buffer type for this backend (for allocating tensors)
    ggml_backend_buffer_type_t buffer_type() const;

    // Execute a computation graph on this backend
    enum ggml_status compute(struct ggml_cgraph* graph);

    // Synchronize (wait for async operations to complete)
    void synchronize();

    // Get backend name for logging
    const char* name() const;

    // Check if this is a GPU backend
    bool is_gpu() const;

    // Static helper: get best available backend type
    static Type best_available();

private:
    ggml_backend_t backend_ = nullptr;
    Type type_ = Type::CPU;
};

// RAII wrapper for ggml_backend_buffer_t
class BackendBuffer {
public:
    BackendBuffer() = default;
    explicit BackendBuffer(ggml_backend_buffer_t buf) : buffer_(buf) {}
    ~BackendBuffer() { if (buffer_) ggml_backend_buffer_free(buffer_); }

    BackendBuffer(const BackendBuffer&) = delete;
    BackendBuffer& operator=(const BackendBuffer&) = delete;

    BackendBuffer(BackendBuffer&& other) noexcept : buffer_(other.buffer_) {
        other.buffer_ = nullptr;
    }
    BackendBuffer& operator=(BackendBuffer&& other) noexcept {
        if (this != &other) {
            if (buffer_) ggml_backend_buffer_free(buffer_);
            buffer_ = other.buffer_;
            other.buffer_ = nullptr;
        }
        return *this;
    }

    ggml_backend_buffer_t handle() const { return buffer_; }
    operator bool() const { return buffer_ != nullptr; }

private:
    ggml_backend_buffer_t buffer_ = nullptr;
};

// Global backend singleton for the application
Backend& global_backend();

// Initialize global backend with specified type
void init_global_backend(Backend::Type type);

}  // namespace dasheng
