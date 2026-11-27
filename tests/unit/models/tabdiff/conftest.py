"""Shared fixtures for TabDiff unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.integration.models.tabdiff.path_helper import TABDIFF_DIR, load_implementation


ASSETS_ROOT = Path(__file__).resolve().parents[3] / "integration" / "assets"

# dataset_name is the Info JSON stem / process_data name.
FIXTURE_ASSETS: dict[str, tuple[Path, str]] = {
    "berka": (ASSETS_ROOT / "tabdiff", "trans"),
    "diabetes": (ASSETS_ROOT / "tabdiff_diabetes", "diabetes"),
}


@pytest.fixture(scope="module")
def tabdiff_dir() -> Path:
    load_implementation()
    return TABDIFF_DIR


@pytest.fixture(scope="module")
def asset_dir() -> Path:
    """Berka asset dir (backward-compatible default for existing unit tests)."""
    return FIXTURE_ASSETS["berka"][0]


@pytest.fixture(scope="module")
def diabetes_asset_dir() -> Path:
    return FIXTURE_ASSETS["diabetes"][0]


def load_asset_info(asset_dir: Path, dataset_name: str | None = None) -> dict:
    """Load Info/<dataset>.json and resolve relative data/encoder paths against asset_dir.

    If ``dataset_name`` is omitted, infer from ``asset_dir`` via ``FIXTURE_ASSETS``,
    defaulting to ``trans`` for the Berka tree.
    """
    if dataset_name is None:
        dataset_name = "trans"
        for _key, (path, name) in FIXTURE_ASSETS.items():
            if asset_dir.resolve() == path.resolve():
                dataset_name = name
                break

    with open(asset_dir / "Info" / f"{dataset_name}.json") as file:
        info = json.load(file)
    info["data_path"] = str(asset_dir / info["data_path"])
    info["test_path"] = str(asset_dir / info["test_path"])
    encoders = info.get("label_encoders_path")
    if encoders and not Path(encoders).is_absolute():
        info["label_encoders_path"] = str(asset_dir / encoders)
    return info
