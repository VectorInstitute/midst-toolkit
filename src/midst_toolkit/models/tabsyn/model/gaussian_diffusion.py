from typing import Any

import torch
from torch import Tensor, nn

from midst_toolkit.models.tabsyn.model.modules import EDMLoss


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
        sigma_min: float = 0,
        sigma_max: float = float("inf"),
        sigma_data: float = 0.5,
    ):
        """Initialize the Precond.

        Args:
            denoise_fn: The denoising function.
            hid_dim: The hidden dimension.
            sigma_min: The minimum supported noise level. Optional, defaults to 0.
            sigma_max: The maximum supported noise level. Optional, defaults to `float("inf")`.
            sigma_data: The expected standard deviation of the training data. Optional, defaults to 0.5.
        """
        super().__init__()

        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.denoise_fn_F = denoise_fn

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        """Forward pass of the Precond.

        Args:
            x: The input data.
            sigma: The sigma.

        Returns:
            The output data.
        """
        x = x.to(torch.float32)

        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        f_x = self.denoise_fn_F((x_in).to(dtype), c_noise.flatten())

        assert f_x.dtype == dtype
        return c_skip * x + c_out * f_x.to(torch.float32)

    def round_sigma(self, sigma: Tensor) -> Tensor:
        """Round the sigma.

        Args:
            sigma: The sigma.

        Returns:
            The rounded sigma.
        """
        return torch.as_tensor(sigma)


class Model(nn.Module):
    def __init__(
        self,
        denoise_fn: nn.Module,
        hid_dim: int,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_data: float = 0.5,
        gamma: float = 5,
        opts: dict[str, Any] | None = None,
        pfgmpp: bool = False,
    ):
        """Initialize the GaussianDiffusion.

        Args:
            denoise_fn: The denoising function.
            hid_dim: The hidden dimension.
            p_mean: The mean of the noise. Optional, defaults to -1.2.
            p_std: The standard deviation of the noise. Optional, defaults to 1.2.
            sigma_data: The standard deviation of the data. Optional, defaults to 0.5.
            gamma: The gamma parameter. Optional, defaults to 5.
            opts: The options. Optional, defaults to None.
            pfgmpp: Whether to use the PFGMPP model. Optional, defaults to False.
        """
        super().__init__()

        self.denoise_fn_d = Precond(denoise_fn, hid_dim)
        self.loss_fn = EDMLoss(p_mean, p_std, sigma_data, hid_dim=hid_dim, gamma=5, opts=None)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the GaussianDiffusion.

        Args:
            x: The input data.

        Returns:
            The loss.
        """
        loss = self.loss_fn(self.denoise_fn_d, x)
        return loss.mean(-1).mean()
