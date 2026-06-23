#include "pipeline.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>

#include "content_adapter.h"
#include "dit.h"
#include "gguf_util.h"
#include "scheduler.h"
#include "t5_encoder.h"
#include "tokenizer.h"
#include "vocoder.h"

namespace dasheng {

namespace {

constexpr int kContentDim = 1024;
constexpr int kLatentDim = 1280;

}  // namespace

struct Pipeline::Impl {
    explicit Impl(const PipelineConfig &config)
        : config(config),
          tokenizer(config.spiece_model_path),
          t5(config.t5_gguf_path, config.n_threads),
          dit_model(config.dit_gguf_path),
          adapter(dit_model, config.n_threads),
          dit(config.dit_gguf_path, config.n_threads),
          vocoder(config.vocoder_gguf_path, config.n_threads) {}

    PipelineConfig config;
    T5Tokenizer tokenizer;
    T5Encoder t5;
    GgufModel dit_model;
    ContentAdapter adapter;
    DiT dit;
    Vocoder vocoder;
};

Pipeline::Pipeline(const PipelineConfig &config) : impl_(new Impl(config)) {}

Pipeline::~Pipeline() { delete impl_; }

std::vector<float> Pipeline::generate(const std::string &prompt, float duration_seconds) {
    (void)duration_seconds;  // Duration comes from the model's prediction

    // 1. Tokenize
    std::vector<int32_t> token_ids = impl_->tokenizer.encode(prompt);
    const int seq_len = static_cast<int>(token_ids.size());
    std::fprintf(stderr, "[pipeline] tokenized prompt: %d tokens\n", seq_len);

    // 2. T5 encode
    std::vector<float> t5_hidden = impl_->t5.encode(token_ids);
    std::fprintf(stderr, "[pipeline] T5 encoded: %d x %d\n", seq_len, impl_->t5.d_model());

    // 3. Content adapter
    ContentAdapterOutput ca_out = impl_->adapter.run(t5_hidden, seq_len);
    const int T = ca_out.global_latent_length;
    std::fprintf(stderr, "[pipeline] content adapter: context_len=%d, latent_length=%d\n",
                 ca_out.context_len, T);

    // 4. Initialize latent noise
    std::vector<float> latent(static_cast<size_t>(T) * kLatentDim);
    {
        unsigned int seed = impl_->config.seed;
        if (seed == 0) {
            seed = static_cast<unsigned int>(std::random_device{}());
        }
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (float &v : latent) {
            v = dist(rng);
        }
        std::fprintf(stderr, "[pipeline] using seed: %u\n", seed);
    }

    // 5. Scheduler
    FlowMatchScheduler scheduler(impl_->config.num_steps, impl_->config.sway_sampling_coef);
    std::vector<float> sigmas = scheduler.sigmas();
    std::vector<float> timesteps = scheduler.timesteps();

    const bool use_cfg = impl_->config.guidance_scale > 1.0f;
    const float cfg_scale = impl_->config.guidance_scale;

    // Pre-compute unconditional context (zeros) for CFG
    std::vector<float> uncond_context(ca_out.context.size(), 0.0f);
    std::vector<float> uncond_ta_content(ca_out.time_aligned_content.size(), 0.0f);

    std::fprintf(stderr, "[pipeline] starting diffusion loop: %d steps, cfg=%s (scale=%.1f)\n",
                 impl_->config.num_steps, use_cfg ? "true" : "false", cfg_scale);

    // 6. Diffusion loop
    for (int step = 0; step < impl_->config.num_steps; ++step) {
        float sigma = sigmas[step];
        float sigma_next = sigmas[step + 1];
        float timestep = timesteps[step];

        std::vector<float> velocity;

        if (use_cfg) {
            // Unconditional forward
            std::vector<float> vel_uncond = impl_->dit.forward(
                latent, T, timestep, uncond_context, ca_out.context_len, uncond_ta_content);

            // Conditional forward
            std::vector<float> vel_cond = impl_->dit.forward(
                latent, T, timestep, ca_out.context, ca_out.context_len, ca_out.time_aligned_content);

            // CFG: v = v_uncond + scale * (v_cond - v_uncond)
            velocity.resize(vel_cond.size());
            for (size_t i = 0; i < velocity.size(); ++i) {
                velocity[i] = vel_uncond[i] + cfg_scale * (vel_cond[i] - vel_uncond[i]);
            }
        } else {
            velocity = impl_->dit.forward(
                latent, T, timestep, ca_out.context, ca_out.context_len, ca_out.time_aligned_content);
        }

        // Euler step: latent = latent + (sigma_next - sigma) * velocity
        float dt = sigma_next - sigma;
        for (size_t i = 0; i < latent.size(); ++i) {
            latent[i] += dt * velocity[i];
        }

        if ((step + 1) % 5 == 0 || step == 0) {
            std::fprintf(stderr, "[pipeline] step %d/%d, sigma=%.4f\n",
                         step + 1, impl_->config.num_steps, sigma);
        }
    }

    std::fprintf(stderr, "[pipeline] diffusion complete, decoding vocoder...\n");

    // 7. Vocoder decode
    std::vector<float> waveform = impl_->vocoder.decode(latent, T);

    std::fprintf(stderr, "[pipeline] generated %zu samples (%.2f seconds at 16kHz)\n",
                 waveform.size(), static_cast<float>(waveform.size()) / 16000.0f);

    return waveform;
}

}  // namespace dasheng
