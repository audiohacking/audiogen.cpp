#pragma once

#include <vector>

// Flow-matching Euler ODE scheduler with sway-sampling sigma schedule.
namespace dasheng {

class FlowMatchScheduler {
public:
    explicit FlowMatchScheduler(int num_steps, float sway_coef = -1.0f);

    // Sigma schedule for the configured number of Euler steps, descending
    // from 1.0 to 0.0 with sway-sampling skew applied.
    std::vector<float> sigmas() const;

    // Timesteps for the DiT (sigmas * 1000).
    std::vector<float> timesteps() const;

    int num_steps() const { return num_steps_; }

private:
    int num_steps_;
    float sway_coef_;
};

}  // namespace dasheng
