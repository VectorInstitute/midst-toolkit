import pandas as pd
import pytest

from midst_toolkit.data_processing.midst_data_processing import load_midst_data_with_test
from midst_toolkit.evaluation.privacy.distance_closest_record import NormType
from midst_toolkit.evaluation.privacy.distance_preprocess import preprocess
from midst_toolkit.evaluation.privacy.nearest_neighbor_distance_ratio import (
    NearestNeighborDistanceRatio,
)


REAL_DATA = pd.DataFrame(
    {
        "column_a": [1, 2, 3, 4, 5],
        "column_b": [4, 5, 6, 7, 8],
        "column_c": ["horse", "dog", "horse", "cat", "cat"],
        "column_d": [-1, -2, -3, -2, -5],
    }
)

SYNTHETIC_DATA = pd.DataFrame(
    {
        "column_a": [1, 2, 3, 4, 5],
        "column_b": [4, 6, 6, -1, 1],
        "column_c": ["cat", "dog", "horse", "cat", "cat"],
        "column_d": [-1, -2, -3, -2, -5],
    }
)

HOLDOUT_DATA = pd.DataFrame(
    {
        "column_a": [2, 3, 4, 5, 6],
        "column_b": [4, 5, 6, 2, 3],
        "column_c": ["cat", "dog", "horse", "cat", "cat"],
        "column_d": [-1, -2, -3, -2, -5],
    }
)

META_INFO = {
    "num_col_idx": [0, 1, 3],
    "cat_col_idx": [2],
}


SYNTHETIC_DATA_PATH = "tests/assets/synthetic_data_dcr.csv"
REAL_DATA_TRAIN_PATH = "tests/assets/real_data_dcr.csv"
REAL_DATA_TEST_PATH = "tests/assets/real_data_test.csv"
META_INFO_PATH = "tests/assets/meta_info.json"


def test_nndr_score_small_data() -> None:
    filtered_real_data = REAL_DATA[["column_a", "column_b", "column_d"]]
    filtered_synthetic_data = SYNTHETIC_DATA[["column_a", "column_b", "column_d"]]
    filtered_holdout_data = HOLDOUT_DATA[["column_a", "column_b", "column_d"]]

    nndr_metric = NearestNeighborDistanceRatio()
    results = nndr_metric.compute(filtered_real_data, filtered_synthetic_data)
    assert len(results) == 2
    target = ((0 / 1.7321) + (1.0000 / 1.4142) + (0 / 1.7321) + (5.9161 / 6.3246) + (5.7446 / 5.8310)) / 5.0
    assert pytest.approx(results["mean_nndr"], abs=1e-5) == target

    results = nndr_metric.compute(filtered_real_data, filtered_synthetic_data, filtered_holdout_data)
    target_holdout = ((1 / 2.4495) + (1.4142 / 2.2361) + (1 / 1.4142) + (3.1623 / 5.3852) + (2.2361 / 3.1623)) / 5.0
    assert len(results) == 4
    assert pytest.approx(results["privacy_loss"], abs=1e-5) == target - target_holdout


def test_nndr_score_small_data_l1() -> None:
    filtered_real_data = REAL_DATA[["column_a", "column_b", "column_d"]]
    filtered_synthetic_data = SYNTHETIC_DATA[["column_a", "column_b", "column_d"]]
    filtered_holdout_data = HOLDOUT_DATA[["column_a", "column_b", "column_d"]]

    nndr_metric = NearestNeighborDistanceRatio(norm=NormType.L1)
    results = nndr_metric.compute(filtered_real_data, filtered_synthetic_data)
    assert len(results) == 2
    target = ((0 / 3) + (1 / 2) + (0 / 3) + (8 / 8) + (7 / 9)) / 5.0
    assert pytest.approx(results["mean_nndr"], abs=1e-5) == target

    results = nndr_metric.compute(filtered_real_data, filtered_synthetic_data, filtered_holdout_data)
    target_holdout = ((1 / 4) + (2 / 3) + (1 / 2) + (4 / 7) + (3 / 4)) / 5.0
    assert len(results) == 4
    assert pytest.approx(results["privacy_loss"], abs=1e-5) == target - target_holdout


def test_nndr_score_small_data_with_categoricals_ordinal() -> None:
    mapped_real_data = REAL_DATA.replace({"cat": 0, "horse": 1, "dog": 2})
    mapped_synthetic_data = SYNTHETIC_DATA.replace({"cat": 0, "horse": 1, "dog": 2})
    mapped_holdout_data = HOLDOUT_DATA.replace({"cat": 0, "horse": 1, "dog": 2})

    nndr_metric = NearestNeighborDistanceRatio()
    results = nndr_metric.compute(mapped_real_data, mapped_synthetic_data)
    assert len(results) == 2
    target = ((1.0000 / 2.6458) + (1.0000 / 1.7321) + (0 / 2.0000) + (6.0000 / 6.6332) + (5.8310 / 6.1644)) / 5.0
    assert pytest.approx(results["mean_nndr"], abs=1e-5) == target

    results = nndr_metric.compute(mapped_real_data, mapped_synthetic_data, mapped_holdout_data)
    target_holdout = (
        (1.0000 / 3.1623) + (1.4142 / 2.4495) + (1.0000 / 1.7321) + (3.1623 / 5.3852) + (2.2361 / 3.1623)
    ) / 5.0
    assert len(results) == 4
    assert pytest.approx(results["privacy_loss"], abs=1e-5) == target - target_holdout


def test_nndr_score_small_data_with_categoricals_one_hot() -> None:
    synthetic_data, real_data, holdout_data = preprocess(META_INFO, SYNTHETIC_DATA, REAL_DATA, HOLDOUT_DATA)

    nndr_metric = NearestNeighborDistanceRatio()
    results = nndr_metric.compute(real_data, synthetic_data)
    assert len(results) == 2
    target = ((1.0897 / 1.4142) + (0.2500 / 1.4577) + (0 / 0.8660) + (2.0000 / 2.0463) + (1.6956 / 1.7500)) / 5.0
    assert pytest.approx(results["mean_nndr"], abs=1e-4) == target

    results = nndr_metric.compute(real_data, synthetic_data, holdout_data)
    target_holdout = (
        (0.2500 / 1.1456) + (0.3536 / 1.5207) + (0.2500 / 1.4577) + (0.7906 / 1.3463) + (0.5590 / 0.7906)
    ) / 5.0
    assert len(results) == 4
    assert pytest.approx(results["privacy_loss"], abs=1e-5) == target - target_holdout


def test_nndr_score() -> None:
    real_data, synthetic_data, holdout_data, meta_info = load_midst_data_with_test(
        REAL_DATA_TRAIN_PATH, SYNTHETIC_DATA_PATH, META_INFO_PATH, REAL_DATA_TEST_PATH
    )

    synthetic_data, real_data, holdout_data = preprocess(meta_info, synthetic_data, real_data, holdout_data)
    nndr_metric = NearestNeighborDistanceRatio()
    results = nndr_metric.compute(real_data, synthetic_data, holdout_data)
    assert pytest.approx(results["mean_nndr"], abs=1e-8) == 0.9782823717907417
    assert pytest.approx(results["privacy_loss"], abs=1e-8) == 0.005370743246908338


def test_nndr_score_with_preprocess() -> None:
    real_data, synthetic_data, holdout_data, meta_info = load_midst_data_with_test(
        REAL_DATA_TRAIN_PATH, SYNTHETIC_DATA_PATH, META_INFO_PATH, REAL_DATA_TEST_PATH
    )

    # Preprocessing internally should return the same result
    nndr_metric = NearestNeighborDistanceRatio(meta_info=meta_info, do_preprocess=True)
    results = nndr_metric.compute(real_data, synthetic_data, holdout_data)
    assert pytest.approx(results["mean_nndr"], abs=1e-8) == 0.9782823717907417
    assert pytest.approx(results["privacy_loss"], abs=1e-8) == 0.005370743246908338
