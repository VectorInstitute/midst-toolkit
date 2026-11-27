"""TabSyn-style public path: preprocess → train → reload checkpoint → sample CSV.

Unlike the golden suite (in-process train+sample), this locks Trainer(ckpt_path=...)
reload then sample_synthetic → CSV, matching tabsyn's test_train_load_and_synthesize.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from tests.integration.models.tabdiff import golden_runner
from tests.integration.models.tabdiff.path_helper import load_implementation


FIXTURE_KEYS = tuple(golden_runner.FIXTURES)
NUM_SAMPLES = 8


@pytest.fixture
def scratch_dir(tmp_path: Path) -> Path:
    work = tmp_path / "tabdiff_train_reload"
    work.mkdir()
    yield work
    shutil.rmtree(work, ignore_errors=True)


def _build_train_stack(processed_dir: Path, fixture: golden_runner.Fixture, ckpt_dir: Path, result_dir: Path):
    """Construct diffusion + Trainer from processed fixture data (CPU, golden protocol flags)."""
    load_implementation()
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP
    from tabdiff.trainer import Trainer
    from utils_train import TabDiffDataset

    import src

    impl_dir = load_implementation()
    device = torch.device("cpu")
    with open(processed_dir / "info.json") as file:
        info = __import__("json").load(file)

    golden_runner.set_all_seeds(golden_runner.SEED)
    train_data = TabDiffDataset(
        fixture.dataset_name,
        str(processed_dir),
        info,
        y_only=False,
        isTrain=True,
        dequant_dist="none",
        int_dequant_factor=0,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=golden_runner.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    d_numerical, categories = train_data.d_numerical, train_data.categories

    raw_config = src.load_config(str(impl_dir / "tabdiff" / "configs" / "tabdiff_configs.toml"))
    raw_config["unimodmlp_params"]["d_numerical"] = d_numerical
    raw_config["unimodmlp_params"]["categories"] = (categories + 1).tolist()
    raw_config["diffusion_params"]["scheduler"] = "power_mean_per_column"
    raw_config["diffusion_params"]["cat_scheduler"] = "log_linear_per_column"
    raw_config["diffusion_params"]["sampler_params"]["stochastic_sampler"] = False
    raw_config["diffusion_params"]["sampler_params"]["second_order_correction"] = False

    backbone = UniModMLP(**raw_config["unimodmlp_params"])
    model = Model(backbone, **raw_config["diffusion_params"]["edm_params"])
    model.to(device)
    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        y_only_model=None,
        **raw_config["diffusion_params"],
        device=device,
    )
    diffusion.to(device)

    train_cfg = dict(raw_config["train"]["main"])
    train_cfg["steps"] = golden_runner.NUM_TRAIN_EPOCHS
    train_cfg["batch_size"] = golden_runner.BATCH_SIZE
    # Fire once on the last epoch so model_{epoch}.pt is written.
    train_cfg["check_val_every"] = golden_runner.NUM_TRAIN_EPOCHS

    trainer = Trainer(
        diffusion,
        train_loader,
        train_data,
        train_data,
        golden_runner._NoMetrics(info),
        golden_runner.RecordingLogger(),
        **train_cfg,
        sample_batch_size=NUM_SAMPLES,
        num_samples_to_generate=NUM_SAMPLES,
        model_save_path=str(ckpt_dir),
        result_save_path=str(result_dir),
        device=device,
        ckpt_path=None,
        y_only=False,
    )
    return trainer, raw_config, info, d_numerical, categories, device


@pytest.mark.integration_test()
@pytest.mark.parametrize("fixture_key", FIXTURE_KEYS)
def test_train_reload_checkpoint_and_synthesize(scratch_dir: Path, fixture_key: str) -> None:
    """Mirrors tabsyn test_train_load_and_synthesize: train, reload ckpt, sample CSV."""
    fixture = golden_runner.FIXTURES[fixture_key]
    processed_dir = golden_runner.run_preprocess(scratch_dir, fixture)

    train_ckpt = scratch_dir / "train_ckpt"
    train_result = scratch_dir / "train_result"
    train_ckpt.mkdir()
    train_result.mkdir()

    trainer, raw_config, info, d_numerical, categories, device = _build_train_stack(
        processed_dir, fixture, train_ckpt, train_result
    )
    # evaluate_generation needs full metrics; stub so check_val_every only exercises save.
    trainer.evaluate_generation = lambda **kwargs: ({}, {}, None)  # type: ignore[method-assign]
    trainer.run_loop()

    model_ckpts = sorted(train_ckpt.glob("model_*.pt"))
    assert model_ckpts, "short train with check_val_every=steps must write model_{epoch}.pt"
    ckpt_path = model_ckpts[-1]

    # Fresh stack + Trainer(ckpt_path=...) — the reload path TabSyn exercises.
    reload_ckpt_dir = scratch_dir / "reload_ckpt"
    reload_result = scratch_dir / "reload_result"
    reload_ckpt_dir.mkdir()
    reload_result.mkdir()

    load_implementation()
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP
    from tabdiff.trainer import Trainer
    from utils_train import TabDiffDataset

    golden_runner.set_all_seeds(golden_runner.SEED)
    train_data = TabDiffDataset(
        fixture.dataset_name,
        str(processed_dir),
        info,
        y_only=False,
        isTrain=True,
        dequant_dist="none",
        int_dequant_factor=0,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=golden_runner.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    backbone = UniModMLP(**raw_config["unimodmlp_params"])
    model = Model(backbone, **raw_config["diffusion_params"]["edm_params"])
    model.to(device)
    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        y_only_model=None,
        **raw_config["diffusion_params"],
        device=device,
    )
    diffusion.to(device)

    train_cfg = dict(raw_config["train"]["main"])
    train_cfg["steps"] = 1
    train_cfg["batch_size"] = golden_runner.BATCH_SIZE
    train_cfg["check_val_every"] = 999

    reloaded = Trainer(
        diffusion,
        train_loader,
        train_data,
        train_data,
        golden_runner._NoMetrics(info),
        golden_runner.RecordingLogger(),
        **train_cfg,
        sample_batch_size=NUM_SAMPLES,
        num_samples_to_generate=NUM_SAMPLES,
        model_save_path=str(reload_ckpt_dir),
        result_save_path=str(reload_result),
        device=device,
        ckpt_path=str(ckpt_path),
        y_only=False,
    )

    golden_runner.set_all_seeds(golden_runner.SEED)
    samples = reloaded.sample_synthetic(NUM_SAMPLES, keep_nan_samples=True, ema=False)
    assert isinstance(samples, pd.DataFrame)

    out_csv = reload_result / f"{fixture.dataset_name}_synthetic.csv"
    samples.to_csv(out_csv, index=False)

    assert out_csv.is_file()
    loaded = pd.read_csv(out_csv)
    assert len(loaded) == NUM_SAMPLES
    assert list(loaded.columns) == list(samples.columns)
    assert not loaded.isna().any().any()
    # Decoded columns should match info column_names when present.
    expected_cols = info.get("column_names")
    if expected_cols:
        assert list(loaded.columns) == list(expected_cols)
