#pragma once

#include <cstdio>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

namespace dasheng {

// Backend pair: primary (GPU if available) + CPU fallback
struct BackendPair {
    ggml_backend_t backend;      // Primary backend (Metal/CUDA/CPU)
    ggml_backend_t cpu_backend;  // CPU fallback (nullptr if primary is CPU)
    ggml_backend_sched_t sched;  // Scheduler (optional, created separately)
    bool has_gpu;                // True if primary is GPU
};

// Initialize backends. Returns cached pair on subsequent calls.
// name is used for logging.
BackendPair backend_init(const char* name);

// Create a scheduler for the backend pair with given graph size.
ggml_backend_sched_t backend_sched_new(const BackendPair& bp, size_t graph_size);

// Release backends (for cleanup)
void backend_release(ggml_backend_t backend, ggml_backend_t cpu_backend);

// Get global backend pair (initializes on first call)
BackendPair& global_backend_pair();

// Weight context for staging tensor data before GPU transfer
// Following acestep.cpp pattern
struct WeightCtx {
    struct ggml_context* ctx;
    ggml_backend_buffer_t buffer;

    struct PendingCopy {
        ggml_tensor* tensor;
        const void* src;
        size_t size;
        size_t offset;
    };
    std::vector<PendingCopy> pending;
};

// Initialize weight context for N tensors
inline bool wctx_init(WeightCtx* wctx, size_t n_tensors) {
    size_t ctx_size = ggml_tensor_overhead() * n_tensors + 1024;
    struct ggml_init_params params = {ctx_size, nullptr, true};  // no_alloc
    wctx->ctx = ggml_init(params);
    wctx->buffer = nullptr;
    wctx->pending.clear();
    return wctx->ctx != nullptr;
}

// Allocate backend buffer and copy all pending data
inline bool wctx_alloc(WeightCtx* wctx, ggml_backend_t backend) {
    if (!wctx->ctx) return false;

    // Allocate buffer for all context tensors on this backend
    wctx->buffer = ggml_backend_alloc_ctx_tensors(wctx->ctx, backend);
    if (!wctx->buffer) {
        return false;
    }

    // Mark as weights for scheduler optimization
    ggml_backend_buffer_set_usage(wctx->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Copy all pending data to backend
    size_t total_bytes = 0;
    for (const auto& p : wctx->pending) {
        ggml_backend_tensor_set(p.tensor, p.src, p.offset, p.size);
        total_bytes += p.size;
    }

    std::fprintf(stderr, "[weights] transferred %.1f MB to %s\n",
                 total_bytes / (1024.0 * 1024.0), ggml_backend_name(backend));

    wctx->pending.clear();
    return true;
}

// Free weight context
inline void wctx_free(WeightCtx* wctx) {
    if (wctx->buffer) {
        ggml_backend_buffer_free(wctx->buffer);
        wctx->buffer = nullptr;
    }
    if (wctx->ctx) {
        ggml_free(wctx->ctx);
        wctx->ctx = nullptr;
    }
    wctx->pending.clear();
}

}  // namespace dasheng
