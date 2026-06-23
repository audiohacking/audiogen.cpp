#include "backend.h"

#include <cstdio>
#include <stdexcept>

#include "ggml-cpu.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

namespace dasheng {

namespace {

std::unique_ptr<Backend> g_backend;

}  // namespace

Backend::Type Backend::best_available() {
#ifdef GGML_USE_METAL
    return Type::Metal;
#elif defined(GGML_USE_CUDA)
    return Type::CUDA;
#else
    return Type::CPU;
#endif
}

Backend::Backend(Type type) {
    if (type == Type::Auto) {
        type = best_available();
    }

    type_ = type;

    switch (type) {
        case Type::Metal:
#ifdef GGML_USE_METAL
            backend_ = ggml_backend_metal_init();
            if (!backend_) {
                std::fprintf(stderr, "[backend] Metal init failed, falling back to CPU\n");
                type_ = Type::CPU;
                backend_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
            }
#else
            std::fprintf(stderr, "[backend] Metal not compiled in, using CPU\n");
            type_ = Type::CPU;
            backend_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
#endif
            break;

        case Type::CUDA:
#ifdef GGML_USE_CUDA
            backend_ = ggml_backend_cuda_init(0);  // device 0
            if (!backend_) {
                std::fprintf(stderr, "[backend] CUDA init failed, falling back to CPU\n");
                type_ = Type::CPU;
                backend_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
            }
#else
            std::fprintf(stderr, "[backend] CUDA not compiled in, using CPU\n");
            type_ = Type::CPU;
            backend_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
#endif
            break;

        case Type::CPU:
        case Type::Auto:
        default:
            backend_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
            type_ = Type::CPU;
            break;
    }

    if (!backend_) {
        throw std::runtime_error("Failed to initialize any backend");
    }

    std::fprintf(stderr, "[backend] initialized: %s\n", name());
}

Backend::~Backend() {
    if (backend_) {
        ggml_backend_free(backend_);
    }
}

ggml_backend_buffer_type_t Backend::buffer_type() const {
    return ggml_backend_get_default_buffer_type(backend_);
}

enum ggml_status Backend::compute(struct ggml_cgraph* graph) {
    return ggml_backend_graph_compute(backend_, graph);
}

void Backend::synchronize() {
    ggml_backend_synchronize(backend_);
}

const char* Backend::name() const {
    return ggml_backend_name(backend_);
}

bool Backend::is_gpu() const {
    return type_ == Type::Metal || type_ == Type::CUDA;
}

Backend& global_backend() {
    if (!g_backend) {
        g_backend = std::make_unique<Backend>(Backend::Type::Auto);
    }
    return *g_backend;
}

void init_global_backend(Backend::Type type) {
    g_backend = std::make_unique<Backend>(type);
}

}  // namespace dasheng
