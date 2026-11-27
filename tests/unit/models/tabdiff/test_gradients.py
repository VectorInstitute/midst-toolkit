"""Gradient participation checks for TabDiff diffusion training step."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def test_mixed_loss_produces_finite_grads_for_denoise_and_schedules(tabdiff_dir: Path) -> None:
    """A nondegenerate batch should backprop into denoise_fn and learnable schedules."""
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    device = torch.device("cpu")
    categories = np.array([3, 4], dtype=np.int64)
    d_numerical = 2
    backbone = UniModMLP(
        num_layers=2,
        d_token=4,
        n_head=1,
        factor=32,
        bias=True,
        dim_t=64,
        use_mlp=True,
        d_numerical=d_numerical,
        categories=(categories + 1).tolist(),
    )
    model = Model(backbone, precond=True, sigma_data=1.0, net_conditioning="sigma")
    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        y_only_model=None,
        num_timesteps=4,
        scheduler="power_mean_per_column",
        cat_scheduler="log_linear_per_column",
        noise_dist="uniform_t",
        edm_params={"precond": True, "sigma_data": 1.0, "net_conditioning": "sigma"},
        noise_dist_params={"P_mean": -1.2, "P_std": 1.2},
        noise_schedule_params={
            "sigma_min": 0.002,
            "sigma_max": 80,
            "rho": 7,
            "eps_max": 1e-3,
            "eps_min": 1e-5,
            "rho_init": 7.0,
            "rho_offset": 5.0,
            "k_init": -6.0,
            "k_offset": 1.0,
        },
        sampler_params={"stochastic_sampler": False, "second_order_correction": False},
        device=device,
    )
    diffusion.to(device)
    diffusion.train()

    torch.manual_seed(0)
    x_num = torch.randn(8, d_numerical, device=device)
    x_cat = torch.stack(
        [torch.randint(0, int(k), (8,), device=device) for k in categories],
        dim=1,
    ).float()
    x = torch.cat([x_num, x_cat], dim=1)

    dloss, closs = diffusion.mixed_loss(x)
    loss = dloss + closs
    assert torch.isfinite(loss)
    loss.backward()

    denoise_grads = [p.grad for p in diffusion._denoise_fn.parameters() if p.requires_grad]
    assert denoise_grads, "denoise_fn has no trainable parameters"
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in denoise_grads)

    # Per-column schedules register learnable rho_raw / k_raw.
    assert diffusion.num_schedule.rho_raw.grad is not None
    assert torch.isfinite(diffusion.num_schedule.rho_raw.grad).all()
    assert diffusion.cat_schedule.k_raw.grad is not None
    assert torch.isfinite(diffusion.cat_schedule.k_raw.grad).all()
