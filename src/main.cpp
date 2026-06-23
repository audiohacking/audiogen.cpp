#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <vector>

#include "backend.h"
#include "content_adapter.h"
#include "dit.h"
#include "pipeline.h"
#include "vocoder.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf_util.h"
#include "t5_encoder.h"
#include "tokenizer.h"

namespace {

// Compose prompt from individual fields, matching the HuggingFace API
std::string compose_prompt(const std::string &caption,
                            const std::string &speech = "",
                            const std::string &asr = "",
                            const std::string &sfx = "",
                            const std::string &music = "",
                            const std::string &env = "") {
    if (caption.empty()) {
        throw std::runtime_error("caption is required and cannot be empty");
    }
    std::string result;
    // Tag order matches reference implementation
    const std::pair<const char *, const std::string *> tags[] = {
        {"<|caption|>", &caption},
        {"<|speech|>", &speech},
        {"<|asr|>", &asr},
        {"<|sfx|>", &sfx},
        {"<|music|>", &music},
        {"<|env|>", &env},
    };
    for (const auto &[tag, value] : tags) {
        if (!value->empty()) {
            if (!result.empty()) result += " ";
            result += tag;
            result += " ";
            result += *value;
        }
    }
    return result;
}

// Parse a simple JSON-like batch file (one prompt per line or JSON array)
std::vector<std::string> parse_batch_file(const std::string &path) {
    std::vector<std::string> prompts;
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("failed to open batch file: " + path);
    }
    std::string line;
    while (std::getline(f, line)) {
        // Skip empty lines and comments
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        if (line[start] == '#') continue;
        // Trim
        size_t end = line.find_last_not_of(" \t\r\n");
        if (end != std::string::npos) {
            prompts.push_back(line.substr(start, end - start + 1));
        }
    }
    return prompts;
}

std::vector<float> read_binary_f32(const char *path, size_t count) {
    std::vector<float> out(count);
    FILE *f = std::fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string("failed to open ") + path);
    size_t n = std::fread(out.data(), sizeof(float), count, f);
    std::fclose(f);
    if (n != count) throw std::runtime_error(std::string("short read from ") + path);
    return out;
}
}  // namespace

// Phase 0 smoke test: build a tiny ggml graph (a + b) on the CPU backend to
// confirm the vendored ggml submodule links and runs correctly. Replaced by
// the real CLI (src/pipeline.h) once the model graphs are implemented.
static int run_smoke_test() {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        return 1;
    }

    struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    for (int i = 0; i < 4; ++i) {
        ((float *) a->data)[i] = static_cast<float>(i);
        ((float *) b->data)[i] = static_cast<float>(10 * i);
    }
    struct ggml_tensor *sum = ggml_add(ctx, a, b);

    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, sum);
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    std::printf("ggml smoke test: [");
    for (int i = 0; i < 4; ++i) {
        std::printf("%s%.1f", i ? ", " : "", ((float *) sum->data)[i]);
    }
    std::printf("]\n");

    ggml_free(ctx);
    return 0;
}

// Phase 4 validation harness: encode a fixed token-id sequence with the T5
// encoder and dump hidden states so they can be diffed against a PyTorch
// reference run on the same ids. Not part of the eventual CLI.
static int run_test_t5(const char *gguf_path, int argc, char **argv) {
    std::vector<int32_t> ids;
    for (int i = 0; i < argc; ++i) {
        ids.push_back(std::atoi(argv[i]));
    }
    dasheng::T5Encoder enc(gguf_path);
    std::vector<float> hidden = enc.encode(ids);
    std::printf("d_model=%d seq_len=%zu\n", enc.d_model(), ids.size());
    for (float v : hidden) {
        std::printf("%.6f\n", v);
    }
    return 0;
}

// Stage 1 validation harness: T5-encode a fixed prompt, run it through
// content_encoder.text_encoder.proj + CrossAttentionAdapter, and dump the
// resulting context / global latent length / time_aligned_content so they
// can be diffed against the PyTorch reference dump
// (dump_stage1_reference.py).
static int run_test_content_adapter(const char *t5_gguf_path, const char *dit_gguf_path,
                                     const char *spiece_path, const char *prompt) {
    dasheng::T5Tokenizer tok(spiece_path);
    std::vector<int32_t> ids = tok.encode(prompt);

    dasheng::T5Encoder t5(t5_gguf_path);
    std::vector<float> hidden = t5.encode(ids);

    dasheng::GgufModel dit_model(dit_gguf_path);
    dasheng::ContentAdapter adapter(dit_model);
    dasheng::ContentAdapterOutput out = adapter.run(hidden, static_cast<int>(ids.size()));

    std::printf("seq_len=%zu\n", ids.size());
    std::printf("global_latent_length=%d\n", out.global_latent_length);
    std::printf("context[0][:8]=");
    for (int i = 0; i < 8; ++i) std::printf("%s%.6f", i ? "," : "", out.context[i]);
    std::printf("\n");
    std::printf("context_sum=%.6f\n", std::accumulate(out.context.begin(), out.context.end(), 0.0));
    return 0;
}

// Stage 2 validation harness: run the DiT backbone forward pass on a fixed
// latent/context/time_aligned_content (dumped from the PyTorch reference by
// dump_stage2_dit_reference.py) and dump the predicted velocity so it can be
// diffed numerically against stage2_dit_ref.npz.
static int run_test_dit(const char *dit_gguf_path, const char *latent_bin, const char *context_bin,
                         const char *ta_content_bin, int T, int context_len, float timestep) {
    dasheng::DiT dit(dit_gguf_path);
    std::vector<float> latent = read_binary_f32(latent_bin, static_cast<size_t>(T) * dit.latent_dim());
    std::vector<float> context = read_binary_f32(context_bin, static_cast<size_t>(context_len) * 1024);
    std::vector<float> ta_content = read_binary_f32(ta_content_bin, static_cast<size_t>(T) * 1024);

    std::vector<float> out = dit.forward(latent, T, timestep, context, context_len, ta_content);

    std::printf("T=%d context_len=%d latent_dim=%d\n", T, context_len, dit.latent_dim());
    std::printf("out[0][:8]=");
    for (int i = 0; i < 8; ++i) std::printf("%s%.6f", i ? "," : "", out[i]);
    std::printf("\n");
    std::printf("out_sum=%.6f\n", std::accumulate(out.begin(), out.end(), 0.0));

    FILE *f = std::fopen("/tmp/dit_test_actual_output.bin", "wb");
    std::fwrite(out.data(), sizeof(float), out.size(), f);
    std::fclose(f);
    return 0;
}

// Stage 3 validation harness: run the Vocoder (upsampler + Vocos decoder +
// ISTFT) on a fixed latent (the same one used for the stage-2 DiT test) and
// dump the reconstructed waveform so it can be diffed numerically against
// stage3_vocoder_ref.npz.
static int run_test_vocoder(const char *vocoder_gguf_path, const char *latent_bin, int T) {
    dasheng::Vocoder vocoder(vocoder_gguf_path);
    std::vector<float> latent = read_binary_f32(latent_bin, static_cast<size_t>(T) * 1280);

    std::vector<float> out = vocoder.decode(latent, T);

    std::printf("T=%d out_len=%zu\n", T, out.size());
    std::printf("out[:8]=");
    for (int i = 0; i < 8; ++i) std::printf("%s%.8f", i ? "," : "", out[i]);
    std::printf("\n");
    std::printf("out_sum=%.6f\n", std::accumulate(out.begin(), out.end(), 0.0));

    FILE *f = std::fopen("/tmp/vocoder_test_actual_output.bin", "wb");
    std::fwrite(out.data(), sizeof(float), out.size(), f);
    std::fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && std::strcmp(argv[1], "--smoke-test") == 0) {
        return run_smoke_test();
    }
    if (argc > 2 && std::strcmp(argv[1], "--test-t5") == 0) {
        return run_test_t5(argv[2], argc - 3, argv + 3);
    }
    if (argc > 5 && std::strcmp(argv[1], "--test-content-adapter") == 0) {
        return run_test_content_adapter(argv[2], argv[3], argv[4], argv[5]);
    }
    if (argc > 8 && std::strcmp(argv[1], "--test-dit") == 0) {
        return run_test_dit(argv[2], argv[3], argv[4], argv[5], std::atoi(argv[6]), std::atoi(argv[7]),
                             static_cast<float>(std::atof(argv[8])));
    }
    if (argc > 4 && std::strcmp(argv[1], "--test-vocoder") == 0) {
        return run_test_vocoder(argv[2], argv[3], std::atoi(argv[4]));
    }

    // Full pipeline CLI
    if (argc < 5) {
        std::fprintf(stderr,
                     "Usage: dasheng-audiogen <t5.gguf> <dit.gguf> <vocoder.gguf> <spiece.model> [options]\n"
                     "\nPrompt options (use ONE of these):\n"
                     "  --prompt TEXT      Raw prompt with <|caption|> prefix\n"
                     "  --caption TEXT     Caption/description (auto-prefixed with <|caption|>)\n"
                     "  --batch FILE       Batch file with one prompt per line\n"
                     "\nOptional prompt modifiers (combine with --caption):\n"
                     "  --speech TEXT      Speech content (<|speech|> tag)\n"
                     "  --asr TEXT         ASR transcript (<|asr|> tag)\n"
                     "  --sfx TEXT         Sound effects (<|sfx|> tag)\n"
                     "  --music TEXT       Music description (<|music|> tag)\n"
                     "  --env TEXT         Environment/ambience (<|env|> tag)\n"
                     "\nGeneration options:\n"
                     "  --steps N          Number of diffusion steps (default: 25)\n"
                     "  --duration SECS    Audio duration in seconds (default: 10)\n"
                     "  --cfg SCALE        Classifier-free guidance scale (default: 3.0)\n"
                     "  --sway COEF        Sway sampling coefficient (default: -1.0)\n"
                     "  --seed N           Random seed (default: random)\n"
                     "  --threads N        Number of threads (default: 4)\n"
                     "\nOutput options:\n"
                     "  --output FILE      Output WAV file (default: output.wav)\n"
                     "  --output-dir DIR   Output directory for batch mode (default: .)\n"
                     "\nTest modes:\n"
                     "  --smoke-test                    Verify ggml build\n"
                     "  --test-t5 <gguf> [ids...]       Test T5 encoder\n"
                     "  --test-content-adapter ...      Test content adapter\n"
                     "  --test-dit ...                  Test DiT backbone\n"
                     "  --test-vocoder ...              Test vocoder\n"
                     "\nExamples:\n"
                     "  dasheng-audiogen t5.gguf dit.gguf vocoder.gguf spiece.model --caption \"A dog barking\"\n"
                     "  dasheng-audiogen t5.gguf dit.gguf vocoder.gguf spiece.model --caption \"Piano\" --music \"soft jazz\"\n"
                     "  dasheng-audiogen t5.gguf dit.gguf vocoder.gguf spiece.model --batch prompts.txt --output-dir out/\n");
        return 1;
    }

    // Parse arguments
    const char *t5_path = argv[1];
    const char *dit_path = argv[2];
    const char *vocoder_path = argv[3];
    const char *spiece_path = argv[4];

    std::string prompt;
    std::string caption;
    std::string speech;
    std::string asr;
    std::string sfx;
    std::string music;
    std::string env;
    std::string batch_file;
    std::string output_path = "output.wav";
    std::string output_dir = ".";
    int num_steps = 25;
    float duration_seconds = 10.0f;
    float cfg_scale = 3.0f;
    float sway_coef = -1.0f;
    int n_threads = 4;
    unsigned int seed = 0;
    bool seed_set = false;

    for (int i = 5; i < argc; ++i) {
        if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "--caption") == 0 && i + 1 < argc) {
            caption = argv[++i];
        } else if (std::strcmp(argv[i], "--speech") == 0 && i + 1 < argc) {
            speech = argv[++i];
        } else if (std::strcmp(argv[i], "--asr") == 0 && i + 1 < argc) {
            asr = argv[++i];
        } else if (std::strcmp(argv[i], "--sfx") == 0 && i + 1 < argc) {
            sfx = argv[++i];
        } else if (std::strcmp(argv[i], "--music") == 0 && i + 1 < argc) {
            music = argv[++i];
        } else if (std::strcmp(argv[i], "--env") == 0 && i + 1 < argc) {
            env = argv[++i];
        } else if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch_file = argv[++i];
        } else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            num_steps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--duration") == 0 && i + 1 < argc) {
            duration_seconds = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(argv[i], "--cfg") == 0 && i + 1 < argc) {
            cfg_scale = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(argv[i], "--sway") == 0 && i + 1 < argc) {
            sway_coef = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = static_cast<unsigned int>(std::atoi(argv[++i]));
            seed_set = true;
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (std::strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            n_threads = std::atoi(argv[++i]);
        }
    }

    // Build prompts list
    std::vector<std::string> prompts;
    if (!batch_file.empty()) {
        prompts = parse_batch_file(batch_file);
        if (prompts.empty()) {
            std::fprintf(stderr, "Error: batch file is empty\n");
            return 1;
        }
    } else if (!prompt.empty()) {
        // Raw prompt - must start with <|caption|>
        if (prompt.find("<|caption|>") != 0) {
            std::fprintf(stderr, "Error: --prompt must start with <|caption|>\n");
            return 1;
        }
        prompts.push_back(prompt);
    } else if (!caption.empty()) {
        // Compose from individual fields
        prompts.push_back(compose_prompt(caption, speech, asr, sfx, music, env));
    } else {
        // Default prompt
        prompts.push_back("<|caption|> A dog barking");
    }

    // Initialize backend (auto-selects Metal on macOS, CUDA if available, else CPU)
    dasheng::BackendPair& bp = dasheng::global_backend_pair();

    std::fprintf(stderr, "dasheng-audiogen\n");
    std::fprintf(stderr, "  backend:  %s%s\n", ggml_backend_name(bp.backend), bp.has_gpu ? " (GPU)" : "");
    std::fprintf(stderr, "  t5:       %s\n", t5_path);
    std::fprintf(stderr, "  dit:      %s\n", dit_path);
    std::fprintf(stderr, "  vocoder:  %s\n", vocoder_path);
    std::fprintf(stderr, "  spiece:   %s\n", spiece_path);
    std::fprintf(stderr, "  prompts:  %zu\n", prompts.size());
    std::fprintf(stderr, "  steps:    %d\n", num_steps);
    std::fprintf(stderr, "  cfg:      %.1f\n", cfg_scale);
    std::fprintf(stderr, "  sway:     %.1f\n", sway_coef);
    std::fprintf(stderr, "  threads:  %d\n", n_threads);
    if (prompts.size() == 1) {
        std::fprintf(stderr, "  prompt:   %s\n", prompts[0].c_str());
        std::fprintf(stderr, "  output:   %s\n", output_path.c_str());
    } else {
        std::fprintf(stderr, "  output:   %s/\n", output_dir.c_str());
    }

    dasheng::PipelineConfig config;
    config.t5_gguf_path = t5_path;
    config.dit_gguf_path = dit_path;
    config.vocoder_gguf_path = vocoder_path;
    config.spiece_model_path = spiece_path;
    config.num_steps = num_steps;
    config.guidance_scale = cfg_scale;
    config.sway_sampling_coef = sway_coef;
    config.duration_seconds = duration_seconds;
    config.n_threads = n_threads;
    if (seed_set) {
        config.seed = seed;
    }

    dasheng::Pipeline pipeline(config);

    // Helper to write WAV file
    auto write_wav = [](const std::string &path, const std::vector<float> &waveform) -> bool {
        FILE *wav = std::fopen(path.c_str(), "wb");
        if (!wav) return false;

        const uint32_t sample_rate = 16000;
        const uint16_t bits_per_sample = 16;
        const uint16_t num_channels = 1;
        const uint32_t num_samples = static_cast<uint32_t>(waveform.size());
        const uint32_t data_size = num_samples * num_channels * bits_per_sample / 8;
        const uint32_t file_size = 36 + data_size;

        // WAV header
        std::fwrite("RIFF", 1, 4, wav);
        std::fwrite(&file_size, 4, 1, wav);
        std::fwrite("WAVE", 1, 4, wav);
        std::fwrite("fmt ", 1, 4, wav);
        uint32_t fmt_size = 16;
        std::fwrite(&fmt_size, 4, 1, wav);
        uint16_t audio_format = 1;  // PCM
        std::fwrite(&audio_format, 2, 1, wav);
        std::fwrite(&num_channels, 2, 1, wav);
        std::fwrite(&sample_rate, 4, 1, wav);
        uint32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
        std::fwrite(&byte_rate, 4, 1, wav);
        uint16_t block_align = num_channels * bits_per_sample / 8;
        std::fwrite(&block_align, 2, 1, wav);
        std::fwrite(&bits_per_sample, 2, 1, wav);
        std::fwrite("data", 1, 4, wav);
        std::fwrite(&data_size, 4, 1, wav);

        // Convert float samples to 16-bit PCM
        for (float sample : waveform) {
            sample = std::max(-1.0f, std::min(1.0f, sample));
            int16_t pcm = static_cast<int16_t>(sample * 32767.0f);
            std::fwrite(&pcm, 2, 1, wav);
        }

        std::fclose(wav);
        return true;
    };

    // Process prompts
    for (size_t i = 0; i < prompts.size(); ++i) {
        const std::string &p = prompts[i];
        std::fprintf(stderr, "\n[%zu/%zu] Generating: %s\n", i + 1, prompts.size(),
                     p.length() > 60 ? (p.substr(0, 57) + "...").c_str() : p.c_str());

        std::vector<float> waveform = pipeline.generate(p, duration_seconds);

        // Determine output path
        std::string out;
        if (prompts.size() == 1) {
            out = output_path;
        } else {
            std::ostringstream oss;
            oss << output_dir << "/output_" << std::setfill('0') << std::setw(4) << i << ".wav";
            out = oss.str();
        }

        if (!write_wav(out, waveform)) {
            std::fprintf(stderr, "Failed to write: %s\n", out.c_str());
            return 1;
        }
        std::fprintf(stderr, "Saved: %s (%.2f seconds)\n", out.c_str(),
                     static_cast<float>(waveform.size()) / 16000.0f);
    }

    std::fprintf(stderr, "\nDone. Generated %zu audio file(s).\n", prompts.size());
    return 0;
}
