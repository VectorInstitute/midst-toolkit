from collections.abc import Callable

import torch
from torch import Tensor, nn


ModuleType = str | Callable[..., nn.Module]


class SiLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the SiLU.

        Args:
            x: The input data.

        Returns:
            The output data.
        """
        return x * torch.sigmoid(x)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels: int, max_positions: int = 10000, endpoint: bool = False):
        """Initialize the PositionalEmbedding.

        Args:
            num_channels: The number of channels.
            max_positions: The maximum positions. Optional, defaults to 10000.
            endpoint: Whether to include the endpoint. Optional, defaults to False.
        """
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the PositionalEmbedding.

        Args:
            x: The input data.

        Returns:
            The output data.
        """
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


class MLPDiffusion(nn.Module):
    def __init__(self, d_in: int, dim_t: int = 512):
        """Initialize the MLPDiffusion.

        Args:
            d_in: The input dimension.
            dim_t: The dimension of the time embedding. Optional, defaults to 512.
        """
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t))

    def forward(self, x: Tensor, noise_labels: Tensor, class_labels: Tensor | None = None) -> Tensor:
        """Forward pass of the MLPDiffusion.

        Args:
            x: The input data.
            noise_labels: The noise labels.
            class_labels: The class labels. Optional, defaults to None.

        Returns:
            The output data.
        """
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        emb = self.time_embed(emb)

        x = self.proj(x) + emb
        return self.mlp(x)


class Preconditioner(nn.Module):
    def __init__(
        self,
        denoise_model: nn.Module,
        hidden_dimension: int,
        sigma_min: float = 0,
        sigma_max: float = float("inf"),
        sigma_data: float = 0.5,
    ):
        """Initialize the Precond.

        Args:
            denoise_model: The denoising model.
            hidden_dimension: The hidden dimension.
            sigma_min: The minimum supported noise level. Optional, defaults to 0.
            sigma_max: The maximum supported noise level. Optional, defaults to `float("inf")`.
            sigma_data: The expected standard deviation of the training data. Optional, defaults to 0.5.
        """
        super().__init__()

        self.hidden_dimension = hidden_dimension
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.denoise_model = denoise_model

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

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        f_x = self.denoise_model((x_in).to(torch.float32), c_noise.flatten())

        assert f_x.dtype == torch.float32
        return c_skip * x + c_out * f_x.to(torch.float32)


class EDMLoss:
    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_data: float = 0.5,
        hidden_dimension: int = 100,
    ) -> None:
        """Initialize the EDMLoss.

        Args:
            p_mean: The mean of the noise. Optional, defaults to -1.2.
            p_std: The standard deviation of the noise. Optional, defaults to 1.2.
            sigma_data: The standard deviation of the data. Optional, defaults to 0.5.
            hidden_dimension: The hidden dimension. Optional, defaults to 100.
        """
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data
        self.hidden_dimension = hidden_dimension

    def __call__(self, denoise_fn: nn.Module, data: Tensor) -> Tensor:
        """Calculate the loss.

        Args:
            denoise_fn: The denoising function.
            data: The input data.

        Returns:
            The loss.
        """
        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()

        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        y = data
        n = torch.randn_like(y) * sigma.unsqueeze(1)
        d_yn = denoise_fn(y + n, sigma)

        target = y
        return weight.unsqueeze(1) * ((d_yn - target) ** 2)


class Model(nn.Module):
    def __init__(
        self,
        denoise_model: nn.Module,
        hidden_dimension: int,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_data: float = 0.5,
    ):
        """Initialize the model.

        Args:
            denoise_model: The denoising model.
            hidden_dimension: The hidden dimension.
            p_mean: The mean of the noise. Optional, defaults to -1.2.
            p_std: The standard deviation of the noise. Optional, defaults to 1.2.
            sigma_data: The standard deviation of the data. Optional, defaults to 0.5.
        """
        super().__init__()

        self.preconditioner = Preconditioner(denoise_model, hidden_dimension)
        self.loss_fn = EDMLoss(p_mean, p_std, sigma_data, hidden_dimension=hidden_dimension)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: The input data.

        Returns:
            The mean loss.
        """
        loss = self.loss_fn(self.preconditioner, x)
        return loss.mean(-1).mean()
