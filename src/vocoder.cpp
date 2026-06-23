#include "vocoder.h"

#include <cmath>
#include <complex>
#include <cstring>
#include <stdexcept>

#include "ggml-cpu.h"
#include "ggml.h"
#include "pocketfft_hdronly.h"

namespace dasheng {

namespace {

constexpr int kDim = 1280;          // decoder hidden dim == latent_dim here
constexpr int kNumConvNeXt = 12;
constexpr int kUpsampleStride = 2;  // ConvTranspose1d kernel == stride
constexpr float kLnEps = 1e-6f;     // Vocos LayerNorms use eps=1e-6

}  // namespace

struct Vocoder::Impl {
    explicit Impl(const std::string &gguf_path, int n_threads) : model(gguf_path), n_threads(n_threads) {}
    GgufModel model;
    int n_threads;
};

Vocoder::Vocoder(const std::string &gguf_path, int n_threads) : impl_(new Impl(gguf_path, n_threads)) {}

Vocoder::~Vocoder() { delete impl_; }

std::vector<float> Vocoder::decode(const std::vector<float> &latent, int T) {
    GgufModel &model = impl_->model;

    const int n_fft = static_cast<int>(model.kv_i32("dasheng_audiogen.istft_n_fft"));     // 1280
    const int hop = static_cast<int>(model.kv_i32("dasheng_audiogen.istft_hop"));         // 320
    const int n_freq = n_fft / 2 + 1;                                                     // 641
    const int T2 = (T - 1) * kUpsampleStride + kUpsampleStride;                            // 2*T

    const size_t mem_size = 512ull * 1024 * 1024 + static_cast<size_t>(T2) * 1800000ull;
    struct ggml_init_params cparams = {mem_size, nullptr, false};
    struct ggml_context *ctx = ggml_init(cparams);
    if (!ctx) {
        throw std::runtime_error("Vocoder::decode: ggml_init failed");
    }

    auto linear = [&](struct ggml_tensor *w, struct ggml_tensor *b, struct ggml_tensor *in) {
        struct ggml_tensor *out = ggml_mul_mat(ctx, w, in);
        if (b) out = ggml_add(ctx, out, b);
        return out;
    };
    auto layer_norm = [&](struct ggml_tensor *in, const std::string &prefix) {
        struct ggml_tensor *normed = ggml_norm(ctx, in, kLnEps);
        normed = ggml_mul(ctx, normed, model.tensor(prefix + ".weight"));
        normed = ggml_add(ctx, normed, model.tensor(prefix + ".bias"));
        return normed;
    };
    // conv1d with kernel>1: data must be transposed to [length, channels] before
    // ggml_conv_1d and transposed back to [channels, length] after.
    auto conv1d = [&](struct ggml_tensor *kernel, struct ggml_tensor *x_cl, int stride, int pad) {
        const int64_t L = x_cl->ne[1];
        struct ggml_tensor *x_lc = ggml_cont(ctx, ggml_transpose(ctx, x_cl));  // [L, C]
        x_lc = ggml_reshape_3d(ctx, x_lc, L, x_cl->ne[0], 1);
        struct ggml_tensor *out = ggml_conv_1d(ctx, kernel, x_lc, stride, pad, 1);  // [OL, OC, 1]
        out = ggml_reshape_2d(ctx, out, out->ne[0], out->ne[1]);
        return ggml_cont(ctx, ggml_transpose(ctx, out));  // [OC, OL]
    };

    // --- input ---
    struct ggml_tensor *x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kDim, T);
    std::memcpy(x_in->data, latent.data(), latent.size() * sizeof(float));

    // --- upsampler: ConvTranspose1d(kDim,kDim,k=2,s=2,p=0) ---
    struct ggml_tensor *x_lc = ggml_cont(ctx, ggml_transpose(ctx, x_in));  // [T, kDim]
    struct ggml_tensor *up = ggml_conv_transpose_1d(ctx, model.tensor("upsampler.weight"), x_lc, kUpsampleStride, 0, 1);  // [T2, kDim, 1]
    up = ggml_reshape_2d(ctx, up, up->ne[0], up->ne[1]);
    up = ggml_cont(ctx, ggml_transpose(ctx, up));  // [kDim, T2]
    up = ggml_add(ctx, up, model.tensor("upsampler.bias"));

    // --- backbone.embed: Conv1d(kDim,kDim,k=7,p=3) ---
    struct ggml_tensor *x = conv1d(model.tensor("decoder.backbone.embed.weight"), up, 1, 3);
    x = ggml_add(ctx, x, model.tensor("decoder.backbone.embed.bias"));
    x = layer_norm(x, "decoder.backbone.norm");

    for (int i = 0; i < kNumConvNeXt; ++i) {
        const std::string prefix = "decoder.backbone.convnext." + std::to_string(i);
        struct ggml_tensor *residual = x;

        // depthwise Conv1d(kDim,kDim,k=7,p=3,groups=kDim), implemented as a sum
        // of 7 per-channel-scaled shifts (ggml_conv_1d_dw is flagged unreliable
        // in ggml.h; this formulation only uses well-tested ops).
        struct ggml_tensor *dw_w =
            ggml_reshape_2d(ctx, ggml_cast(ctx, model.tensor(prefix + ".dwconv.weight"), GGML_TYPE_F32), 7, kDim);
        dw_w = ggml_cont(ctx, ggml_transpose(ctx, dw_w));  // [kDim, 7]
        struct ggml_tensor *padded = ggml_pad_ext(ctx, x, 0, 0, 3, 3, 0, 0, 0, 0);  // [kDim, T2+6]
        struct ggml_tensor *dw_out = nullptr;
        for (int k = 0; k < 7; ++k) {
            struct ggml_tensor *shifted =
                ggml_view_2d(ctx, padded, kDim, T2, padded->nb[1], static_cast<size_t>(k) * padded->nb[1]);
            struct ggml_tensor *w_k = ggml_view_1d(ctx, dw_w, kDim, static_cast<size_t>(k) * dw_w->nb[1]);
            struct ggml_tensor *term = ggml_mul(ctx, shifted, w_k);
            dw_out = dw_out ? ggml_add(ctx, dw_out, term) : term;
        }
        dw_out = ggml_add(ctx, dw_out, model.tensor(prefix + ".dwconv.bias"));

        struct ggml_tensor *h = layer_norm(dw_out, prefix + ".norm");
        h = linear(model.tensor(prefix + ".pwconv1.weight"), model.tensor(prefix + ".pwconv1.bias"), h);
        h = ggml_gelu_erf(ctx, h);
        h = linear(model.tensor(prefix + ".pwconv2.weight"), model.tensor(prefix + ".pwconv2.bias"), h);
        h = ggml_mul(ctx, h, model.tensor(prefix + ".gamma"));

        x = ggml_add(ctx, residual, h);
    }

    x = layer_norm(x, "decoder.backbone.final_layer_norm");

    // --- ISTFTHead ---
    struct ggml_tensor *head_out =
        linear(model.tensor("decoder.head.out.weight"), model.tensor("decoder.head.out.bias"), x);  // [n_fft+2, T2]
    struct ggml_tensor *mag = ggml_cont(ctx, ggml_view_2d(ctx, head_out, n_freq, T2, head_out->nb[1], 0));
    struct ggml_tensor *phase = ggml_cont(
        ctx, ggml_view_2d(ctx, head_out, n_freq, T2, head_out->nb[1], static_cast<size_t>(n_freq) * head_out->nb[0]));
    mag = ggml_clamp(ctx, ggml_exp(ctx, mag), -INFINITY, 1e2f);

    struct ggml_tensor *s_real = ggml_mul(ctx, mag, ggml_cos(ctx, phase));
    struct ggml_tensor *s_imag = ggml_mul(ctx, mag, ggml_sin(ctx, phase));

    struct ggml_cgraph *graph = ggml_new_graph_custom(ctx, 1u << 14, false);
    ggml_build_forward_expand(graph, s_real);
    ggml_build_forward_expand(graph, s_imag);
    ggml_graph_compute_with_ctx(ctx, graph, impl_->n_threads);

    std::vector<float> real_host(static_cast<size_t>(n_freq) * T2);
    std::vector<float> imag_host(static_cast<size_t>(n_freq) * T2);
    std::memcpy(real_host.data(), s_real->data, real_host.size() * sizeof(float));
    std::memcpy(imag_host.data(), s_imag->data, imag_host.size() * sizeof(float));

    ggml_free(ctx);

    // torch.hann_window(n_fft, periodic=True) is a constant buffer, not a
    // learned weight, so it isn't stored in the GGUF -- recompute it here.
    std::vector<float> window(n_fft);
    for (int i = 0; i < n_fft; ++i) {
        window[i] = 0.5f - 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * i / n_fft);
    }

    // --- ISTFT (custom "same" padding, see reference/vocos.py ISTFT.forward) ---
    // real_host/imag_host come straight from a ggml tensor with ne0=n_freq
    // (fastest-varying), ne1=T2 -- so the flat index for (f,t) is t*n_freq+f,
    // not f*T2+t.
    std::vector<std::complex<float>> spec(static_cast<size_t>(n_freq) * T2);
    for (int t = 0; t < T2; ++t) {
        for (int f = 0; f < n_freq; ++f) {
            const size_t idx = static_cast<size_t>(t) * n_freq + f;
            spec[static_cast<size_t>(t) * n_freq + f] = std::complex<float>(real_host[idx], imag_host[idx]);
        }
    }

    std::vector<float> ifft(static_cast<size_t>(n_fft) * T2);
    {
        const pocketfft::shape_t shape_out{static_cast<size_t>(n_fft), static_cast<size_t>(T2)};
        const pocketfft::stride_t stride_in{static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)),
                                             static_cast<std::ptrdiff_t>(n_freq * sizeof(std::complex<float>))};
        const pocketfft::stride_t stride_out{static_cast<std::ptrdiff_t>(sizeof(float)),
                                              static_cast<std::ptrdiff_t>(n_fft * sizeof(float))};
        pocketfft::c2r<float>(shape_out, stride_in, stride_out, /*axis=*/0, /*forward=*/false, spec.data(),
                               ifft.data(), 1.0f / static_cast<float>(n_fft));
    }

    for (int t = 0; t < T2; ++t) {
        for (int i = 0; i < n_fft; ++i) {
            ifft[static_cast<size_t>(t) * n_fft + i] *= window[i];
        }
    }

    const int output_size = (T2 - 1) * hop + n_fft;
    std::vector<float> y(output_size, 0.0f);
    std::vector<float> window_envelope(output_size, 0.0f);
    for (int t = 0; t < T2; ++t) {
        const int base = t * hop;
        for (int i = 0; i < n_fft; ++i) {
            y[base + i] += ifft[static_cast<size_t>(t) * n_fft + i];
            window_envelope[base + i] += window[i] * window[i];
        }
    }

    const int pad = (n_fft - hop) / 2;
    std::vector<float> audio(output_size - 2 * pad);
    for (size_t i = 0; i < audio.size(); ++i) {
        audio[i] = y[pad + i] / window_envelope[pad + i];
    }

    return audio;
}

}  // namespace dasheng
