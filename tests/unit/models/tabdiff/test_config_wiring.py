"""Public config options must reach the diffusion components (A5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tests.integration.models.tabdiff.path_helper import TABDIFF_DIR


def _load_raw_config() -> dict:
    import src

    return src.load_config(str(TABDIFF_DIR / "tabdiff" / "configs" / "tabdiff_configs.toml"))


def test_load_config_wires_train_and_diffusion_from_toml(tabdiff_dir: Path) -> None:
    """Shipped TOML must expose the keys needed to build UniModMLP + diffusion + train kwargs."""
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    raw = _load_raw_config()
    for section in ("unimodmlp_params", "diffusion_params", "train", "data", "sample"):
        assert section in raw, f"missing top-level config section {section!r}"
    assert "main" in raw["train"]
    train_main = raw["train"]["main"]
    for key in ("steps", "lr", "batch_size", "check_val_every", "weight_decay"):
        assert key in train_main, f"missing train.main.{key}"

    diffusion_params = raw["diffusion_params"]
    for key in ("num_timesteps", "scheduler", "cat_scheduler", "edm_params", "sampler_params"):
        assert key in diffusion_params, f"missing diffusion_params.{key}"
    for key in ("precond", "sigma_data", "net_conditioning"):
        assert key in diffusion_params["edm_params"], f"missing edm_params.{key}"

    categories = np.array([2, 3], dtype=np.int64)
    d_numerical = 3
    unimod = dict(raw["unimodmlp_params"])
    unimod["d_numerical"] = d_numerical
    unimod["categories"] = (categories + 1).tolist()

    device = torch.device("cpu")
    backbone = UniModMLP(**unimod)
    model = Model(backbone, **diffusion_params["edm_params"])
    # Use TOML schedulers as shipped (not the golden per-column override).
    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        y_only_model=None,
        **diffusion_params,
        device=device,
    )
    assert diffusion.num_schedule is not None
    assert diffusion.cat_schedule is not None
    assert train_main["steps"] > 0
    assert train_main["batch_size"] > 0


def test_scheduler_config_selects_per_column_noise_modules(tabdiff_dir: Path) -> None:
    from tabdiff.models.noise_schedule import LogLinearNoise_PerColumn, PowerMeanNoise_PerColumn
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    raw = _load_raw_config()
    # Default TOML uses non-per-column names; golden/protocol overrides these.
    raw["diffusion_params"]["scheduler"] = "power_mean_per_column"
    raw["diffusion_params"]["cat_scheduler"] = "log_linear_per_column"
    categories = np.array([2, 3], dtype=np.int64)
    d_numerical = 3
    raw["unimodmlp_params"]["d_numerical"] = d_numerical
    raw["unimodmlp_params"]["categories"] = (categories + 1).tolist()

    device = torch.device("cpu")
    backbone = UniModMLP(**raw["unimodmlp_params"])
    model = Model(backbone, **raw["diffusion_params"]["edm_params"])
    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        y_only_model=None,
        **raw["diffusion_params"],
        device=device,
    )
    assert isinstance(diffusion.num_schedule, PowerMeanNoise_PerColumn)
    assert isinstance(diffusion.cat_schedule, LogLinearNoise_PerColumn)
    assert diffusion.num_schedule.rho().shape == (d_numerical,)
    assert diffusion.cat_schedule.k().shape == (len(categories),)
