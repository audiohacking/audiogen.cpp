#include "backend.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <thread>

#include "ggml-cpu.h"

namespace dasheng {

namespace {

// Singleton backend pair
BackendPair g_backend_pair = {nullptr, nullptr, nullptr, false};
bool g_initialized = false;

// Get physical core count (hardware_concurrency includes hyperthreads)
int get_physical_cores() {
    int n = static_cast<int>(std::thread::hardware_concurrency());
    // Assume 2 threads per core (hyperthreading)
    return n > 1 ? n / 2 : 1;
}

}  // namespace

BackendPair backend_init(const char* name) {
    if (g_initialized) {
        return g_backend_pair;
    }

    BackendPair bp = {nullptr, nullptr, nullptr, false};

    // Check for explicit backend selection via environment
    const char* env_backend = std::getenv("GGML_BACKEND");

    // Try to get the best available backend (Metal on macOS, CUDA on Linux, etc.)
    if (env_backend) {
        bp.backend = ggml_backend_init_by_name(env_backend, nullptr);
        if (!bp.backend) {
            std::fprintf(stderr, "[%s] warning: requested backend '%s' not found\n", name, env_backend);
        }
    }

    if (!bp.backend) {
        bp.backend = ggml_backend_init_best();
    }

    if (!bp.backend) {
        std::fprintf(stderr, "[%s] error: no backend available\n", name);
        throw std::runtime_error("No GGML backend available");
    }

    // Check if we got a GPU backend
    ggml_backend_dev_t dev = ggml_backend_get_device(bp.backend);
    if (dev) {
        enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);
        bp.has_gpu = (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU);
    }

    std::fprintf(stderr, "[%s] primary backend: %s%s\n",
                 name, ggml_backend_name(bp.backend),
                 bp.has_gpu ? " (GPU)" : "");

    // Always initialize CPU backend as fallback
    bp.cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!bp.cpu_backend) {
        std::fprintf(stderr, "[%s] warning: CPU backend init failed\n", name);
    } else {
        // Set thread count for CPU backend
        int n_threads = get_physical_cores();

        // Get set_n_threads function from registry
        ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(
            ggml_backend_get_device(bp.cpu_backend));
        if (cpu_reg) {
            auto set_threads = (ggml_backend_set_n_threads_t)
                ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_set_n_threads");
            if (set_threads) {
                set_threads(bp.cpu_backend, n_threads);
                std::fprintf(stderr, "[%s] CPU backend: %d threads\n", name, n_threads);
            }
        }
    }

    // If primary backend is same as CPU, don't duplicate
    if (bp.cpu_backend && bp.backend == bp.cpu_backend) {
        bp.cpu_backend = nullptr;
    }

    g_backend_pair = bp;
    g_initialized = true;

    return bp;
}

ggml_backend_sched_t backend_sched_new(const BackendPair& bp, size_t graph_size) {
    ggml_backend_t backends[3];
    ggml_backend_buffer_type_t buffer_types[3];
    int n_backends = 0;

    // Primary backend first (gets priority)
    backends[n_backends] = bp.backend;
    buffer_types[n_backends] = ggml_backend_get_default_buffer_type(bp.backend);
    n_backends++;

    // CPU fallback if different from primary
    if (bp.cpu_backend && bp.cpu_backend != bp.backend) {
        backends[n_backends] = bp.cpu_backend;
        // Use host buffer type from GPU if available (pinned memory)
        if (bp.has_gpu) {
            ggml_backend_dev_t dev = ggml_backend_get_device(bp.backend);
            ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
            buffer_types[n_backends] = host_buft ? host_buft : ggml_backend_cpu_buffer_type();
        } else {
            buffer_types[n_backends] = ggml_backend_cpu_buffer_type();
        }
        n_backends++;
    }

    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends,
        buffer_types,
        n_backends,
        graph_size,
        false,  // parallel
        true    // op_offload - let scheduler pick best backend per op
    );

    if (!sched) {
        throw std::runtime_error("Failed to create backend scheduler");
    }

    return sched;
}

void backend_release(ggml_backend_t backend, ggml_backend_t cpu_backend) {
    // In a more complete implementation, we'd reference count
    // For now, just mark as uninitialized
    g_initialized = false;

    if (cpu_backend && cpu_backend != backend) {
        ggml_backend_free(cpu_backend);
    }
    if (backend) {
        ggml_backend_free(backend);
    }

    g_backend_pair = {nullptr, nullptr, nullptr, false};
}

BackendPair& global_backend_pair() {
    if (!g_initialized) {
        backend_init("global");
    }
    return g_backend_pair;
}

}  // namespace dasheng
