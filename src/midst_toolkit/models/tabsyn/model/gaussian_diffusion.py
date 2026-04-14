from typing import Any

import torch
from torch import Tensor, nn

from midst_toolkit.models.tabsyn.model.utils import EDMLoss


# ----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

randn_like = torch.randn_like

SIGMA_MIN = 0.002
SIGMA_MAX = 80
rho = 7
S_churn = 1
S_min = 0
S_max = float("inf")
S_noise = 1


class Precond(nn.Module):
    def __init__(
        self,
        denoise_fn: nn.Module,
        hid_dim: int,
        sigma_min: float = 0,  # Minimum supported noise level.
        sigma_max: float = float("inf"),  # Maximum supported noise level.
        sigma_data: float = 0.5,  # Expected standard deviation of the training data.
    ):
        super().__init__()

        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        ###########
        self.denoise_fn_F = denoise_fn

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        x = x.to(torch.float32)

        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        F_x = self.denoise_fn_F((x_in).to(dtype), c_noise.flatten())

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma: Tensor) -> Tensor:
        return torch.as_tensor(sigma)


class Model(nn.Module):
    def __init__(
        self,
        denoise_fn: nn.Module,
        hid_dim: int,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        gamma: float = 5,
        opts: dict[str, Any] | None = None,
        pfgmpp: bool = False,
    ):
        super().__init__()

        self.denoise_fn_D = Precond(denoise_fn, hid_dim)
        self.loss_fn = EDMLoss(P_mean, P_std, sigma_data, hid_dim=hid_dim, gamma=5, opts=None)

    def forward(self, x: Tensor) -> Tensor:
        loss = self.loss_fn(self.denoise_fn_D, x)
        return loss.mean(-1).mean()
