import math
from dataclasses import dataclass

import torch


@dataclass
class SchedulerOutput:
    prev_sample: torch.FloatTensor


class FlowMatchEulerScheduler:

    def __init__(self, num_train_timesteps: int = 1000):
        self.num_train_timesteps = num_train_timesteps
        self.sigmas = None
        self.timesteps = None
        self._step_index = None

    def set_timesteps(self, sigmas, device):
        if isinstance(sigmas, (list, tuple)):
            sigmas = torch.tensor(sigmas, dtype=torch.float32)
        elif not isinstance(sigmas, torch.Tensor):
            sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)

        sigmas = sigmas.to(device=device)
        self.timesteps = sigmas * self.num_train_timesteps
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])
        self._step_index = None

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: torch.FloatTensor,
        sample: torch.FloatTensor,
    ) -> SchedulerOutput:
        if self._step_index is None:
            self._step_index = (self.timesteps == timestep).nonzero()
            self._step_index = 0 if self._step_index.numel() == 0 else self._step_index[0].item()

        sample = sample.to(torch.float32)

        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output
        prev_sample = prev_sample.to(model_output.dtype)

        self._step_index += 1
        return SchedulerOutput(prev_sample=prev_sample)


def compute_sway_sigmas(num_steps: int, sway_sampling_coef: float = -1.0):
    t = torch.linspace(0, 1, num_steps + 1)
    t = t + sway_sampling_coef * (torch.cos(math.pi / 2.0 * t) - 1.0 + t)
    sigmas = 1.0 - t
    return sigmas


def compute_linear_sigmas(num_steps: int):
    return torch.linspace(1.0, 1.0 / num_steps, num_steps)
