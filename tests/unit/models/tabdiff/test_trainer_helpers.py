"""Unit tests for Trainer LR anneal and EMA weight swap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _tiny_trainer(tabdiff_dir: Path, tmp_path: Path):
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP
    from tabdiff.trainer import Trainer

    categories = np.array([3, 4], dtype=np.int64)
    d_numerical = 2
    device = torch.device("cpu")
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

    class _NoMetrics:
        info: dict = {}

    class _Logger:
        def log(self, data: dict) -> None:
            return None

    ckpt_dir = tmp_path / "ckpt"
    result_dir = tmp_path / "result"
    ckpt_dir.mkdir()
    result_dir.mkdir()
    # Empty iterable is enough: these tests never step through data.
    train_iter: list = []
    return Trainer(
        diffusion,
        train_iter,
        dataset=None,
        test_dataset=None,
        metrics=_NoMetrics(),
        logger=_Logger(),
        lr=1e-3,
        weight_decay=0.0,
        steps=10,
        batch_size=4,
        check_val_every=100,
        sample_batch_size=4,
        model_save_path=str(ckpt_dir),
        result_save_path=str(result_dir),
        num_samples_to_generate=4,
        closs_weight_schedule="anneal",
        c_lambda=1.0,
        d_lambda=1.0,
        device=device,
        ckpt_path=None,
        y_only=False,
    )


def test_anneal_lr_scales_with_step_fraction(tabdiff_dir: Path, tmp_path: Path) -> None:
    trainer = _tiny_trainer(tabdiff_dir, tmp_path)
    assert trainer.optimizer.param_groups[0]["lr"] == 1e-3
    trainer._anneal_lr(5)
    assert trainer.optimizer.param_groups[0]["lr"] == 5e-4
    trainer._anneal_lr(10)
    assert trainer.optimizer.param_groups[0]["lr"] == 0.0


def test_to_ema_model_swaps_and_restores_modules(tabdiff_dir: Path, tmp_path: Path) -> None:
    trainer = _tiny_trainer(tabdiff_dir, tmp_path)
    original = trainer.diffusion._denoise_fn
    original_num = trainer.diffusion.num_schedule
    original_cat = trainer.diffusion.cat_schedule
    assert original is not trainer.ema_model

    saved = trainer.to_ema_model()
    assert trainer.diffusion._denoise_fn is trainer.ema_model
    assert trainer.diffusion.num_schedule is trainer.ema_num_schedule
    assert trainer.diffusion.cat_schedule is trainer.ema_cat_schedule

    trainer.to_model(*saved)
    assert trainer.diffusion._denoise_fn is original
    assert trainer.diffusion.num_schedule is original_num
    assert trainer.diffusion.cat_schedule is original_cat
