#include "scheduler.h"

#include <cmath>

namespace dasheng {

FlowMatchScheduler::FlowMatchScheduler(int num_steps, float sway_coef)
    : num_steps_(num_steps), sway_coef_(sway_coef) {}

std::vector<float> FlowMatchScheduler::sigmas() const {
    std::vector<float> result(num_steps_ + 1);
    const float pi_half = static_cast<float>(M_PI) / 2.0f;

    for (int i = 0; i <= num_steps_; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(num_steps_);
        // Sway sampling: t = t + coef * (cos(pi/2 * t) - 1 + t)
        if (sway_coef_ != 0.0f) {
            t = t + sway_coef_ * (std::cos(pi_half * t) - 1.0f + t);
        }
        result[i] = 1.0f - t;
    }
    return result;
}

std::vector<float> FlowMatchScheduler::timesteps() const {
    std::vector<float> sigs = sigmas();
    std::vector<float> result(num_steps_);
    for (int i = 0; i < num_steps_; ++i) {
        result[i] = sigs[i] * 1000.0f;
    }
    return result;
}

}  // namespace dasheng
