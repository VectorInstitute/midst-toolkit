"""Fresh-process sampling from a saved checkpoint bundle (D3)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch


def test_fresh_process_loads_bundle_and_samples(tabdiff_dir: Path, tmp_path: Path) -> None:
    """Sampling after load must work in a new interpreter (not just in-memory)."""
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
    ckpt = tmp_path / "bundle.pt"
    torch.save(
        {
            "denoise_fn": diffusion._denoise_fn.state_dict(),
            "num_schedule": diffusion.num_schedule.state_dict(),
            "cat_schedule": diffusion.cat_schedule.state_dict(),
        },
        ckpt,
    )
    out_npy = tmp_path / "samples.npy"
    script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        import numpy as np
        import torch
        sys.path.insert(0, {str(tabdiff_dir)!r})
        try:
            import icecream  # noqa: F401
        except ImportError:
            import types
            stub = types.ModuleType("icecream")
            stub.install = lambda: None
            sys.modules["icecream"] = stub
        from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
        from tabdiff.modules.main_modules import Model, UniModMLP
        categories = np.array([3, 4], dtype=np.int64)
        d_numerical = 2
        device = torch.device("cpu")
        backbone = UniModMLP(
            num_layers=2, d_token=4, n_head=1, factor=32, bias=True, dim_t=64, use_mlp=True,
            d_numerical=d_numerical, categories=(categories + 1).tolist(),
        )
        model = Model(backbone, precond=True, sigma_data=1.0, net_conditioning="sigma")
        diffusion = UnifiedCtimeDiffusion(
            num_classes=categories, num_numerical_features=d_numerical, denoise_fn=model,
            y_only_model=None, num_timesteps=4,
            scheduler="power_mean_per_column", cat_scheduler="log_linear_per_column",
            noise_dist="uniform_t",
            edm_params={{"precond": True, "sigma_data": 1.0, "net_conditioning": "sigma"}},
            noise_dist_params={{"P_mean": -1.2, "P_std": 1.2}},
            noise_schedule_params={{
                "sigma_min": 0.002, "sigma_max": 80, "rho": 7, "eps_max": 1e-3, "eps_min": 1e-5,
                "rho_init": 7.0, "rho_offset": 5.0, "k_init": -6.0, "k_offset": 1.0,
            }},
            sampler_params={{"stochastic_sampler": False, "second_order_correction": False}},
            device=device,
        )
        state = torch.load({str(ckpt)!r}, map_location=device, weights_only=False)
        diffusion._denoise_fn.load_state_dict(state["denoise_fn"])
        diffusion.num_schedule.load_state_dict(state["num_schedule"])
        diffusion.cat_schedule.load_state_dict(state["cat_schedule"])
        diffusion.eval()
        torch.manual_seed(0)
        samples = diffusion.sample_all(5, batch_size=4, keep_nan_samples=True)
        assert samples.shape[0] == 5
        np.save({str(out_npy)!r}, samples.numpy())
        print("OK", samples.shape)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert out_npy.exists()
    arr = np.load(out_npy)
    assert arr.shape == (5, 4)
