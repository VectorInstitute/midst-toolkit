"""Unit tests for TabDiffDataset shape contracts on Berka and diabetes fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.models.tabdiff.conftest import FIXTURE_ASSETS, load_asset_info


# Prefitted vocab sizes (len(classes_)), not train nunique.
EXPECTED_CATEGORIES = {
    "berka": [3, 6, 9, 14],
    "diabetes": [
        6,
        2,
        9,
        4,
        6,
        11,
        10,
        12,
        17,
        77,
        67,
        74,
        4,
        4,
        4,
        3,
        2,
        1,
        2,
        1,
        2,
        3,
        1,
        2,
        3,
        1,
        1,
        1,
        1,
        1,
        1,
        4,
        2,
        1,
        1,
        1,
        1,
        2,
        2,
        3,
    ],
}


@pytest.fixture(params=list(FIXTURE_ASSETS))
def processed_fixture(
    request: pytest.FixtureRequest,
    tabdiff_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[str, str, Path]:
    """Run process_data for one golden fixture; return (fixture_key, dataset_name, processed_dir)."""
    import os

    import process_dataset

    fixture_key: str = request.param
    asset_dir, dataset_name = FIXTURE_ASSETS[fixture_key]
    work_dir = tmp_path_factory.mktemp(f"tabdiff_dataset_{fixture_key}")
    info_dir = work_dir / "data" / "Info"
    info_dir.mkdir(parents=True)
    (work_dir / "data" / dataset_name).mkdir(parents=True)

    info = load_asset_info(asset_dir, dataset_name)
    with open(info_dir / f"{dataset_name}.json", "w") as file:
        json.dump(info, file, indent=4)

    previous = Path.cwd()
    os.chdir(work_dir)
    try:
        process_dataset.process_data(dataset_name)
    finally:
        os.chdir(previous)
    return fixture_key, dataset_name, work_dir / "data" / dataset_name


def test_tabdiff_dataset_width_and_target_concat(processed_fixture: tuple[str, str, Path]) -> None:
    import utils_train

    fixture_key, dataset_name, processed_dir = processed_fixture
    with open(processed_dir / "info.json") as file:
        info = json.load(file)

    dataset = utils_train.TabDiffDataset(
        dataset_name,
        str(processed_dir),
        info,
        isTrain=True,
        y_only=False,
        dequant_dist="none",
        int_dequant_factor=0.0,
    )

    assert dataset.d_numerical == int(dataset.X.shape[1] - len(dataset.categories))
    assert dataset.X.shape[1] == dataset.d_numerical + len(dataset.categories)
    # Regression: target is folded into the numerical block (num cols + target).
    assert info["task_type"] == "regression"
    assert dataset.d_numerical == len(info["num_col_idx"]) + len(info["target_col_idx"])
    assert len(dataset.categories) == len(info["cat_col_idx"])
    assert list(dataset.categories) == EXPECTED_CATEGORIES[fixture_key]
    assert len(dataset) == info["train_num"]
