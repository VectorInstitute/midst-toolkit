"""Utility functions and constants for the TabSyn model."""

import numpy as np
import torch
from torch import Tensor, randn_like

from midst_toolkit.common.variables import DEVICE
from midst_toolkit.models.tabsyn.model.modules import Preconditioner


SIGMA_MIN = 0.002
SIGMA_MAX = 80
rho = 7
S_churn = 1
S_min = 0
S_max = float("inf")
S_noise = 1


def sample(
    net: Preconditioner, num_samples: int, dim: int, num_steps: int = 50, device: torch.device = DEVICE
) -> Tensor:
    """Sample from the diffusion process.

    Args:
        net: The network.
        num_samples: The number of samples.
        dim: The dimension of the samples.
        num_steps: The number of steps.
        device: The device to use. Optional, defaults to midst_toolkit.common.variables.DEVICE.

    Returns:
        The sampled data.
    """
    latents = torch.randn([num_samples, dim], device=device)

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (
        sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float32) * t_steps[0]

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_next = sample_step(net, num_steps, i, t_cur, t_next, x_next)

    return x_next


def sample_step(net: Preconditioner, num_steps: int, i: int, t_cur: Tensor, t_next: Tensor, x_next: Tensor) -> Tensor:
    """Sample a step of the diffusion process.

    Args:
        net: The network.
        num_steps: The number of steps.
        i: The current step.
        t_cur: The current timestep.
        t_next: The next timestep.
        x_next: The next sample.

    Returns:
        The next sample.
    """
    x_cur = x_next
    # Increase noise temporarily.
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = t_cur + gamma * t_cur
    x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)
    # Euler step.

    denoised = net(x_hat, t_hat).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    # Apply 2nd order correction.
    if i < num_steps - 1:
        denoised = net(x_next, t_next).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next
