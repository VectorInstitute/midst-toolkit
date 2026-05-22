from typing import Any

import torch
from torch import Tensor, nn

from midst_toolkit.models.tabsyn.model.modules import EDMLoss, Precond


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
        """
        super().__init__()

        self.denoise_fn_d = Precond(denoise_fn, hid_dim, sigma_data=sigma_data)
        self.loss_fn = EDMLoss(p_mean, p_std, sigma_data, hid_dim=hid_dim, gamma=gamma, opts=opts)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the GaussianDiffusion.

        Args:
            x: The input data.

        Returns:
            The loss.
        """
        loss = self.loss_fn(self.denoise_fn_d, x)
        return loss.mean(-1).mean()
