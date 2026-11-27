"""Unit tests for TabDiff noise schedules and EMA updates."""

from __future__ import annotations

from pathlib import Path

import torch


def test_power_mean_and_log_linear_endpoints(tabdiff_dir: Path) -> None:
    from tabdiff.models.noise_schedule import LogLinearNoise, PowerMeanNoise

    power = PowerMeanNoise(sigma_min=0.002, sigma_max=80, rho=7)
    log_linear = LogLinearNoise(eps_max=1e-3, eps_min=1e-5)

    t0 = torch.tensor(0.0)
    t1 = torch.tensor(1.0)
    assert torch.isclose(power.total_noise(t0), torch.tensor(0.002), rtol=1e-5, atol=1e-6)
    assert torch.isclose(power.total_noise(t1), torch.tensor(80.0), rtol=1e-5, atol=1e-5)
    # Round-trip through inverse_to_t at an interior sigma.
    sigma = power.total_noise(torch.tensor(0.3))
    assert torch.isclose(power.inverse_to_t(sigma), torch.tensor(0.3), rtol=1e-5, atol=1e-6)

    assert torch.isclose(log_linear.total_noise(t0), -torch.log1p(-torch.tensor(1e-5)), rtol=1e-5, atol=1e-8)
    assert torch.isclose(log_linear.total_noise(t1), -torch.log1p(-torch.tensor(1.0 - 1e-3)), rtol=1e-5, atol=1e-6)


def test_per_column_noise_shapes(tabdiff_dir: Path) -> None:
    from tabdiff.models.noise_schedule import LogLinearNoise_PerColumn, PowerMeanNoise_PerColumn

    num_schedule = PowerMeanNoise_PerColumn(num_numerical=4, rho_init=7.0, rho_offset=5.0)
    cat_schedule = LogLinearNoise_PerColumn(num_categories=3, k_init=-6.0, k_offset=1.0)
    t = torch.rand(8, 1)

    num_sigma = num_schedule.total_noise(t)
    cat_sigma = cat_schedule.total_noise(t)
    assert num_sigma.shape == (8, 4)
    assert cat_sigma.shape == (8, 3)
    assert num_schedule.rho().shape == (4,)
    assert cat_schedule.k().shape == (3,)
    assert torch.all(num_schedule.rho() > 5.0)
    assert torch.all(cat_schedule.k() > 1.0)


def test_update_ema_rate_extremes(tabdiff_dir: Path) -> None:
    import utils_train

    source = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    target_copy = [tensor.clone() for tensor in source]
    target_keep = [torch.zeros_like(tensor) for tensor in source]

    utils_train.update_ema(target_copy, [tensor + 10 for tensor in source], rate=0.0)
    for updated, expected in zip(target_copy, [tensor + 10 for tensor in source], strict=True):
        assert torch.equal(updated, expected)

    utils_train.update_ema(target_keep, source, rate=1.0)
    for updated in target_keep:
        assert torch.equal(updated, torch.zeros_like(updated))
