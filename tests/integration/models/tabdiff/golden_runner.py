"""Run TabDiff preprocessing, training and sampling on tiny fixtures, for golden comparisons.

This module is the only place that imports the implementation under test, so the golden tests can
be pointed at a different implementation by changing the imports here. Golden files are written
with:

    python -m tests.integration.models.tabdiff.golden_runner
    python -m tests.integration.models.tabdiff.golden_runner --dataset berka
    python -m tests.integration.models.tabdiff.golden_runner --dataset diabetes
    python -m tests.integration.models.tabdiff.golden_runner --dataset all
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tests.integration.models.tabdiff.path_helper import load_implementation


ASSETS_ROOT = Path(__file__).resolve().parents[2] / "assets"
SEED = 0
BATCH_SIZE = 32
NUM_TRAIN_EPOCHS = 2
NUM_SAMPLES = 20
PREPROCESS_FILES = (
    "X_num_train.npy",
    "X_cat_train.npy",
    "y_train.npy",
    "X_num_test.npy",
    "X_cat_test.npy",
    "y_test.npy",
    "train.csv",
    "test.csv",
    "info.json",
)
ABSOLUTE_PATH_KEYS = ("data_path", "test_path", "val_path", "label_encoders_path")


@dataclass(frozen=True)
class Fixture:
    """One TabDiff golden fixture (Berka or Diabetes)."""

    key: str
    asset_dir: Path
    dataset_name: str

    @property
    def golden_dir(self) -> Path:
        return self.asset_dir / "golden"


FIXTURES: dict[str, Fixture] = {
    "berka": Fixture(
        key="berka",
        asset_dir=ASSETS_ROOT / "tabdiff",
        dataset_name="trans",
    ),
    "diabetes": Fixture(
        key="diabetes",
        asset_dir=ASSETS_ROOT / "tabdiff_diabetes",
        dataset_name="diabetes",
    ),
}

# Back-compat aliases used by existing tests.
ASSET_DIR = FIXTURES["berka"].asset_dir
GOLDEN_DIR = FIXTURES["berka"].golden_dir
DATASET_NAME = FIXTURES["berka"].dataset_name


class RecordingLogger:
    """Stand-in for the wandb logger that records every logged dictionary."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def define_metric(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log(self, data: dict[str, Any]) -> None:
        self.records.append({key: _to_python(value) for key, value in data.items()})


class _NoMetrics:
    """Stand-in for the evaluation object. Sampling only needs the table info from it."""

    def __init__(self, info: dict[str, Any]) -> None:
        self.info = info


def _to_python(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    return value


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_preprocess(work_dir: Path, fixture: Fixture | None = None) -> Path:
    """Run preprocessing on a fixture inside ``work_dir`` and return the processed data directory."""
    fixture = fixture or FIXTURES["berka"]
    load_implementation()
    import process_dataset

    info_dir = work_dir / "data" / "Info"
    info_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "data" / fixture.dataset_name).mkdir(parents=True, exist_ok=True)

    with open(fixture.asset_dir / "Info" / f"{fixture.dataset_name}.json") as file:
        info = json.load(file)
    info["data_path"] = str(fixture.asset_dir / info["data_path"])
    info["test_path"] = str(fixture.asset_dir / info["test_path"])
    encoders = info.get("label_encoders_path")
    if encoders and not Path(encoders).is_absolute():
        info["label_encoders_path"] = str(fixture.asset_dir / encoders)
    with open(info_dir / f"{fixture.dataset_name}.json", "w") as file:
        json.dump(info, file, indent=4)

    with _working_directory(work_dir):
        process_dataset.process_data(fixture.dataset_name)

    return work_dir / "data" / fixture.dataset_name


def normalize_info(info: dict[str, Any], asset_dir: Path | None = None) -> dict[str, Any]:
    """Make the info file comparable across machines by replacing absolute fixture paths."""
    asset_dir = asset_dir or ASSET_DIR
    normalized = dict(info)
    for key in ABSOLUTE_PATH_KEYS:
        value = normalized.get(key)
        if not value:
            continue
        path = Path(str(value))
        try:
            normalized[key] = str(path.relative_to(asset_dir))
        except ValueError:
            for prefix in ("raw_data/", "data/", "diabetes_test.pkl", "trans_label_encoders.pkl"):
                text = str(value)
                idx = text.find(prefix)
                if idx >= 0:
                    normalized[key] = text[idx:]
                    break
    return normalized


def run_preprocess_outputs(processed_dir: Path, fixture: Fixture | None = None) -> dict[str, Any]:
    """Load the files written by preprocessing, in the form the golden files store them."""
    fixture = fixture or FIXTURES["berka"]
    outputs: dict[str, Any] = {}
    for name in PREPROCESS_FILES:
        path = processed_dir / name
        if name.endswith(".npy"):
            outputs[name] = np.load(path, allow_pickle=True)
        elif name.endswith(".csv"):
            outputs[name] = pd.read_csv(path)
        else:
            with open(path) as file:
                outputs[name] = normalize_info(json.load(file), fixture.asset_dir)
    return outputs


def run_train_and_sample(
    processed_dir: Path,
    output_dir: Path,
    fixture: Fixture | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Train TabDiff for ``NUM_TRAIN_EPOCHS`` epochs and draw ``NUM_SAMPLES`` samples."""
    fixture = fixture or FIXTURES["berka"]
    impl_dir = load_implementation()
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP
    from tabdiff.trainer import Trainer
    from utils_train import TabDiffDataset

    import src

    device = torch.device("cpu")
    with open(processed_dir / "info.json") as file:
        info = json.load(file)

    set_all_seeds(SEED)
    train_data = TabDiffDataset(
        fixture.dataset_name,
        str(processed_dir),
        info,
        y_only=False,
        isTrain=True,
        dequant_dist="none",
        int_dequant_factor=0,
    )
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    d_numerical, categories = train_data.d_numerical, train_data.categories

    raw_config = src.load_config(str(impl_dir / "tabdiff" / "configs" / "tabdiff_configs.toml"))
    raw_config["unimodmlp_params"]["d_numerical"] = d_numerical
    raw_config["unimodmlp_params"]["categories"] = (categories + 1).tolist()
    raw_config["diffusion_params"]["scheduler"] = "power_mean_per_column"
    raw_config["diffusion_params"]["cat_scheduler"] = "log_linear_per_column"
    raw_config["diffusion_params"]["sampler_params"]["stochastic_sampler"] = False
    raw_config["diffusion_params"]["sampler_params"]["second_order_correction"] = False

    ckpt_dir = output_dir / "ckpt"
    result_dir = output_dir / "result"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    raw_config["model_save_path"] = str(ckpt_dir)
    raw_config["result_save_path"] = str(result_dir)

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
    diffusion.train()

    logger = RecordingLogger()
    train_cfg = dict(raw_config["train"]["main"])
    train_cfg["steps"] = NUM_TRAIN_EPOCHS
    train_cfg["batch_size"] = BATCH_SIZE
    train_cfg["check_val_every"] = NUM_TRAIN_EPOCHS + 1

    trainer = Trainer(
        diffusion,
        train_loader,
        train_data,
        train_data,
        _NoMetrics(info),
        logger,
        **train_cfg,
        sample_batch_size=NUM_SAMPLES,
        num_samples_to_generate=NUM_SAMPLES,
        model_save_path=str(ckpt_dir),
        result_save_path=str(result_dir),
        device=device,
        ckpt_path=None,
        y_only=False,
    )
    trainer.run_loop()
    set_all_seeds(SEED)
    samples = trainer.sample_synthetic(NUM_SAMPLES, keep_nan_samples=True, ema=False)
    return logger.records, samples


def write_golden(fixture: Fixture, output_dir: Path | None = None) -> Path:
    """Generate every golden file for one fixture."""
    target = fixture.golden_dir if output_dir is None else output_dir
    scratch = fixture.asset_dir / "_scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    processed_dir = run_preprocess(scratch, fixture)
    preprocess_dir = target / "preprocess"
    training_dir = target / "training"
    sampling_dir = target / "sampling"
    for directory in (preprocess_dir, training_dir, sampling_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for name in PREPROCESS_FILES:
        src_path = processed_dir / name
        if name == "info.json":
            with open(src_path) as file:
                info = normalize_info(json.load(file), fixture.asset_dir)
            with open(preprocess_dir / name, "w") as file:
                json.dump(info, file, indent=4)
        else:
            shutil.copy(src_path, preprocess_dir / name)

    records, samples = run_train_and_sample(processed_dir, scratch / "train", fixture)
    epoch_records = [record for record in records if "epoch" in record]
    with open(training_dir / "losses.json", "w") as file:
        json.dump(epoch_records, file, indent=2)
    samples.to_csv(sampling_dir / "samples.csv", index=False)
    return target


def write_goldens(dataset: str = "all") -> list[Path]:
    """Write goldens for one fixture key or all."""
    if dataset == "all":
        keys = list(FIXTURES)
    else:
        if dataset not in FIXTURES:
            raise SystemExit(f"Unknown dataset {dataset!r}; choose from {sorted(FIXTURES)} or 'all'")
        keys = [dataset]
    written: list[Path] = []
    for key in keys:
        path = write_golden(FIXTURES[key])
        print(f"Golden files written to {path}")
        written.append(path)
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="all",
        choices=[*FIXTURES, "all"],
        help="Which fixture goldens to regenerate (default: all).",
    )
    args = parser.parse_args()
    write_goldens(args.dataset)
