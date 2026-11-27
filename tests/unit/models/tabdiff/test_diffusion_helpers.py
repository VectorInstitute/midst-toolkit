"""Unit tests for UnifiedCtimeDiffusion helpers and seeded loss determinism."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from midst_toolkit.common.random import set_all_random_seeds, unset_all_random_seeds


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
    model.to(device)
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


def test_pad_and_to_one_hot_shapes(tabdiff_dir: Path) -> None:
    device = torch.device("cpu")
    diffusion, categories, _ = _tiny_diffusion(device)
    batch = 5
    x_cat = torch.stack(
        [torch.randint(0, int(k), (batch,)) for k in categories],
        dim=1,
    ).to(device)
    one_hot = diffusion.to_one_hot(x_cat)
    assert one_hot.shape == (batch, int(np.sum(categories + 1)))

    # Fake concatenated class logits matching num_classes_w_mask widths.
    widths = diffusion.num_classes_w_mask
    logits = torch.randn(batch, sum(widths), device=device)
    padded = diffusion.pad(logits, pad_value=-100.0)
    assert padded.shape == (batch, len(widths), max(widths))


def test_mask_index_is_extra_class_not_real_category(tabdiff_dir: Path) -> None:
    """Mask token is num_classes (one past valid cats) and occupies the last one-hot slot."""
    device = torch.device("cpu")
    diffusion, categories, _ = _tiny_diffusion(device)

    assert torch.equal(diffusion.mask_index.cpu(), torch.tensor(categories).long())
    assert diffusion.num_classes_w_mask == tuple(int(c) + 1 for c in categories)

    # Valid category codes are strictly below the mask index.
    valid = torch.tensor([[0, 0], [2, 3]], device=device)
    assert torch.all(valid < diffusion.mask_index)

    # One-hot width includes the mask; placing the mask sets only that slot.
    masked = diffusion.mask_index.unsqueeze(0).to(device)
    one_hot = diffusion.to_one_hot(masked)
    offset = 0
    for width, mask_idx in zip(diffusion.num_classes_w_mask, diffusion.mask_index.tolist(), strict=True):
        slice_ = one_hot[0, offset : offset + width]
        assert int(slice_.argmax()) == int(mask_idx)
        assert float(slice_.sum()) == 1.0
        offset += width

    # Parameterization must keep mask logits at -inf so sampling cannot emit mask as a class.
    batch = 2
    widths = diffusion.num_classes_w_mask
    logits = torch.zeros(batch, sum(widths), device=device)
    xt = torch.zeros(batch, len(categories), dtype=torch.long, device=device)
    out = diffusion._subs_parameterization(logits, xt)
    for i, mask_idx in enumerate(diffusion.mask_index.tolist()):
        assert torch.all(out[:, i, mask_idx] == diffusion.neg_infinity)


def test_q_xt_identity_at_move_chance_zero(tabdiff_dir: Path) -> None:
    device = torch.device("cpu")
    diffusion, categories, _ = _tiny_diffusion(device)
    batch = 4
    x = torch.stack([torch.randint(0, int(k), (batch,)) for k in categories], dim=1).to(device)
    move_chance = torch.zeros(batch, len(categories), device=device)
    xt, _ = diffusion.q_xt(x, move_chance, strategy="hard")
    assert torch.equal(xt, x)


def test_mixed_loss_finite_and_seed_deterministic(tabdiff_dir: Path) -> None:
    device = torch.device("cpu")

    def _loss_once() -> tuple[float, float]:
        set_all_random_seeds(0, use_deterministic_torch_algos=True, disable_torch_benchmarking=True)
        diffusion, categories, d_numerical = _tiny_diffusion(device)
        diffusion.train()
        x_num = torch.randn(8, d_numerical, device=device)
        x_cat = torch.stack(
            [torch.randint(0, int(k), (8,)) for k in categories],
            dim=1,
        ).to(device)
        x = torch.cat([x_num, x_cat.float()], dim=1)
        dloss, closs = diffusion.mixed_loss(x)
        unset_all_random_seeds()
        return float(dloss.detach()), float(closs.detach())

    first = _loss_once()
    second = _loss_once()
    assert np.isfinite(first[0]) and np.isfinite(first[1])
    assert first == second
