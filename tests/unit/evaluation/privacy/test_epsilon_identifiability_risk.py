import pandas as pd
import pytest

from midst_toolkit.data_processing.midst_data_processing import load_midst_data_with_test
from midst_toolkit.evaluation.privacy.distance_preprocess import preprocess
from midst_toolkit.evaluation.privacy.epsilon_identifiability_risk import (
    EpsilonIdentifiabilityNorm,
    EpsilonIdentifiabilityRisk,
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


def test_epsilon_identifiability_risk_small_data_l2() -> None:
    eir_metric = EpsilonIdentifiabilityRisk(
        categorical_columns=[],
        numerical_columns=["column_a", "column_b", "column_d"],
        norm=EpsilonIdentifiabilityNorm.L2,
    )
    results = eir_metric.compute(REAL_DATA, SYNTHETIC_DATA)

    assert len(results) == 1
    target = 3 / 5
    assert pytest.approx(results["epsilon_identifiability_risk"], abs=1e-5) == target

    results = eir_metric.compute(REAL_DATA, SYNTHETIC_DATA, HOLDOUT_DATA)

    target_holdout = 5 / 5
    assert len(results) == 2
    assert pytest.approx(results["privacy_loss"], abs=1e-5) == target - target_holdout

    # Should get the same results even if we include cat columns, since we're using L2
    eir_metric = EpsilonIdentifiabilityRisk(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b", "column_d"],
        norm=EpsilonIdentifiabilityNorm.L2,
    )
    results = eir_metric.compute(REAL_DATA, SYNTHETIC_DATA)

    assert len(results) == 1
    target = 3 / 5
    assert pytest.approx(results["epsilon_identifiability_risk"], abs=1e-5) == target


def test_epsilon_identifiability_risk_small_data_gower() -> None:
    eir_metric = EpsilonIdentifiabilityRisk(
        categorical_columns=[],
        numerical_columns=["column_a", "column_b", "column_d"],
        norm=EpsilonIdentifiabilityNorm.GOWER,
    )
    results = eir_metric.compute(REAL_DATA, SYNTHETIC_DATA)

    assert len(results) == 1
    target = 5 / 5
    assert pytest.approx(results["epsilon_identifiability_risk"], abs=1e-5) == target

    results = eir_metric.compute(REAL_DATA, SYNTHETIC_DATA, HOLDOUT_DATA)

    target_holdout = 5 / 5
    assert len(results) == 2
    assert pytest.approx(results["privacy_loss"], abs=1e-5) == target - target_holdout

    # Using Categorical columns too after preprocess
    real_data, synthetic_data = preprocess(META_INFO, REAL_DATA, SYNTHETIC_DATA)

    eir_metric = EpsilonIdentifiabilityRisk(
        categorical_columns=[3, 4, 5],
        numerical_columns=[0, 1, 2],
        norm=EpsilonIdentifiabilityNorm.GOWER,
    )

    results = eir_metric.compute(real_data, synthetic_data)

    assert len(results) == 1
    target = 4 / 5
    assert pytest.approx(results["epsilon_identifiability_risk"], abs=1e-5) == target


def test_epsilon_identifiability_risk_small_data_with_preprocess() -> None:
    eir_metric = EpsilonIdentifiabilityRisk(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
    )
    results = eir_metric.compute(REAL_DATA, SYNTHETIC_DATA)

    assert len(results) == 1
    target = 3 / 5
    assert pytest.approx(results["epsilon_identifiability_risk"], abs=1e-5) == target

    results = eir_metric.compute(REAL_DATA, SYNTHETIC_DATA, HOLDOUT_DATA)
    target_holdout = 4 / 5

    assert len(results) == 2
    assert pytest.approx(results["privacy_loss"], abs=1e-5) == target - target_holdout


def test_epsilon_identifiability_risk() -> None:
    real_data, synthetic_data, holdout_data, meta_info = load_midst_data_with_test(
        REAL_DATA_TRAIN_PATH, SYNTHETIC_DATA_PATH, META_INFO_PATH, REAL_DATA_TEST_PATH
    )

    synthetic_data, real_data, holdout_data = preprocess(meta_info, synthetic_data, real_data, holdout_data)

    # After one-hot, we'll treat all the categoricals like numerical columns and leave off the target column
    eir_metric = EpsilonIdentifiabilityRisk(
        categorical_columns=[],
        numerical_columns=list(meta_info["cat_col_idx"] + meta_info["num_col_idx"]),
        norm=EpsilonIdentifiabilityNorm.L2,
    )
    results = eir_metric.compute(real_data, synthetic_data, holdout_data)

    assert pytest.approx(results["epsilon_identifiability_risk"], abs=1e-8) == 0.21739130434782608
    assert pytest.approx(results["privacy_loss"], abs=1e-8) == 0.02006688963210701


def test_epsilon_identifiability_risk_with_preprocess() -> None:
    real_data, synthetic_data, holdout_data, meta_info = load_midst_data_with_test(
        REAL_DATA_TRAIN_PATH, SYNTHETIC_DATA_PATH, META_INFO_PATH, REAL_DATA_TEST_PATH
    )
    categorical_columns = [real_data.columns[i] for i in meta_info["cat_col_idx"] + meta_info["target_col_idx"]]
    numerical_columns = [real_data.columns[i] for i in meta_info["num_col_idx"]]
    eir_metric = EpsilonIdentifiabilityRisk(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        do_preprocess=True,
        norm=EpsilonIdentifiabilityNorm.GOWER,
    )
    results = eir_metric.compute(real_data, synthetic_data, holdout_data)

    assert pytest.approx(results["epsilon_identifiability_risk"], abs=1e-8) == 0.46488294314381273
    assert pytest.approx(results["privacy_loss"], abs=1e-8) == 0.023411371237458234
