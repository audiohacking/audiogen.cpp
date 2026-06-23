#pragma once

#include <string>
#include <vector>

namespace dasheng {

struct PipelineConfig {
    std::string t5_gguf_path;
    std::string dit_gguf_path;
    std::string vocoder_gguf_path;
    std::string spiece_model_path;
    int num_steps = 25;
    float guidance_scale = 3.0f;
    float sway_sampling_coef = -1.0f;
    int n_threads = 4;
    unsigned int seed = 0;  // 0 = random seed
};

class Pipeline {
public:
    explicit Pipeline(const PipelineConfig &config);
    ~Pipeline();

    // Generates 16kHz PCM samples for the given text prompt.
    std::vector<float> generate(const std::string &prompt, float duration_seconds);

private:
    struct Impl;
    Impl *impl_;
};

}  // namespace dasheng
