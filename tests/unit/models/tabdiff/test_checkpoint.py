"""Checkpoint key contract, short-run artifact write, and reload sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _tiny_diffusion(device: torch.device):
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP

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
    return diffusion, categories, d_numerical


REQUIRED_CKPT_KEYS = ("denoise_fn", "num_schedule", "cat_schedule")


def test_checkpoint_bundle_requires_denoiser_and_schedules(tabdiff_dir: Path, tmp_path: Path) -> None:
    """Trainer artifacts must carry denoise_fn + both schedule state_dicts (DESIGN.md)."""
    device = torch.device("cpu")
    diffusion, _, _ = _tiny_diffusion(device)
    path = tmp_path / "bundle.pt"
    state_dicts = {
        "denoise_fn": diffusion._denoise_fn.state_dict(),
        "num_schedule": diffusion.num_schedule.state_dict(),
        "cat_schedule": diffusion.cat_schedule.state_dict(),
    }
    torch.save(state_dicts, path)
    loaded = torch.load(path, map_location=device, weights_only=False)
    assert set(REQUIRED_CKPT_KEYS) <= set(loaded.keys())

    fresh, _, _ = _tiny_diffusion(device)
    fresh._denoise_fn.load_state_dict(loaded["denoise_fn"])
    fresh.num_schedule.load_state_dict(loaded["num_schedule"])
    fresh.cat_schedule.load_state_dict(loaded["cat_schedule"])
    for a, b in zip(diffusion._denoise_fn.parameters(), fresh._denoise_fn.parameters(), strict=True):
        assert torch.equal(a.cpu(), b.cpu())
    assert torch.equal(diffusion.num_schedule.rho().cpu(), fresh.num_schedule.rho().cpu())
    assert torch.equal(diffusion.cat_schedule.k().cpu(), fresh.cat_schedule.k().cpu())


def test_short_train_writes_model_ckpt_when_check_val_every_fires(tabdiff_dir: Path, tmp_path: Path) -> None:
    """best_* requires epoch>4000; short runs only get model_{epoch}.pt on check_val_every."""
    from tabdiff.trainer import Trainer

    device = torch.device("cpu")
    diffusion, categories, d_numerical = _tiny_diffusion(device)
    # One tiny batch so one epoch completes.
    x = torch.randn(4, d_numerical + len(categories), device=device)
    x[:, d_numerical] = torch.randint(0, int(categories[0]), (4,), device=device).float()
    x[:, d_numerical + 1] = torch.randint(0, int(categories[1]), (4,), device=device).float()
    train_iter = [x]

    class _Dataset:
        def __init__(self, cats: np.ndarray, features: torch.Tensor, d_num: int) -> None:
            self.categories = cats
            self.X = features
            self.d_numerical = d_num

    class _NoMetrics:
        info: dict = {}

    class _Logger:
        def define_metric(self, *args, **kwargs) -> None:
            return None

        def log(self, data: dict) -> None:
            return None

    ckpt_dir = tmp_path / "ckpt"
    result_dir = tmp_path / "result"
    ckpt_dir.mkdir()
    result_dir.mkdir()
    dataset = _Dataset(categories, x, d_numerical)
    trainer = Trainer(
        diffusion,
        train_iter,
        dataset=dataset,
        test_dataset=dataset,
        metrics=_NoMetrics(),
        logger=_Logger(),
        lr=1e-3,
        weight_decay=0.0,
        steps=1,
        batch_size=4,
        check_val_every=1,
        sample_batch_size=4,
        model_save_path=str(ckpt_dir),
        result_save_path=str(result_dir),
        num_samples_to_generate=4,
        device=device,
        ckpt_path=None,
        y_only=False,
    )
    # evaluate_generation is heavy/requires metrics; stub it for the save path.
    trainer.evaluate_generation = lambda **kwargs: ({}, {}, None)  # type: ignore[method-assign]
    trainer.run_loop()

    model_ckpts = list(ckpt_dir.glob("model_*.pt"))
    assert model_ckpts, "short run with check_val_every=1 must write model_{epoch}.pt"
    assert not list(ckpt_dir.glob("best_model_*.pt")), "best_model_* must not write before epoch 4000"
    loaded = torch.load(model_ckpts[0], map_location=device, weights_only=False)
    assert set(REQUIRED_CKPT_KEYS) <= set(loaded.keys())


def test_sample_all_honors_count_smaller_than_batch(tabdiff_dir: Path) -> None:
    device = torch.device("cpu")
    diffusion, _, _ = _tiny_diffusion(device)
    diffusion.eval()
    torch.manual_seed(0)
    samples = diffusion.sample_all(num_samples=3, batch_size=8, keep_nan_samples=True)
    assert samples.shape[0] == 3
    assert samples.shape[1] == 2 + 2  # d_numerical + n_cat
