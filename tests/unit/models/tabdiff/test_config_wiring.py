"""Public config options must reach the diffusion components (A5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def test_scheduler_config_selects_per_column_noise_modules(tabdiff_dir: Path) -> None:
    from tabdiff.models.noise_schedule import LogLinearNoise_PerColumn, PowerMeanNoise_PerColumn
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

    import src
    from tests.integration.models.tabdiff.path_helper import TABDIFF_DIR

    raw = src.load_config(str(TABDIFF_DIR / "tabdiff" / "configs" / "tabdiff_configs.toml"))
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
