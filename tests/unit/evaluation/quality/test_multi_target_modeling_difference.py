import multiprocessing as mp
import os
from pathlib import Path
from random import choices

import numpy as np
import pandas as pd
import pytest

from midst_toolkit.common.enumerations import ColumnType
from midst_toolkit.common.random import set_all_random_seeds, unset_all_random_seeds
from midst_toolkit.evaluation.quality.multi_target_modeling_difference import MultiTargetModelingDifference


# skip some tests that fail on github due to hanging issues
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def get_regression_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    real_data = pd.DataFrame(
        {
            "column_a": 1.2 * np.random.randn(2500) + 1,
            "column_b": 2.5 * np.random.randn(2500) - 1,
            "column_c": choices(["cat", "horse", "dog"], weights=[0.25, 0.5, 0.25], k=2500),
            "column_d": 1.5 * np.random.randn(2500) + 12,
        }
    )
    # Adding label column as function of other columns
    real_data["column_e"] = real_data["column_a"] + 2 * real_data["column_b"] - 1.1 * real_data["column_d"] + 0.5
    synthetic_data = pd.DataFrame(
        {
            "column_a": 1.1 * np.random.randn(2500) + 0.5,
            "column_b": 2.2 * np.random.randn(2500) - 1,
            "column_c": choices(["cat", "horse", "dog"], weights=[0.35, 0.35, 0.3], k=2500),
            "column_d": 1.5 * np.random.randn(2500) + 10,
        }
    )
    synthetic_data["column_e"] = (
        synthetic_data["column_a"] + 2 * synthetic_data["column_b"] - 1 * synthetic_data["column_d"] + 0.2
    )

    holdout_data = pd.DataFrame(
        {
            "column_a": 1.2 * np.random.randn(2500) + 1,
            "column_b": 2.5 * np.random.randn(2500) - 1,
            "column_c": choices(["cat", "horse", "dog"], weights=[0.25, 0.5, 0.25], k=2500),
            "column_d": 1.5 * np.random.randn(2500) + 12,
        }
    )
    # Adding label column as function of other columns
    holdout_data["column_e"] = (
        holdout_data["column_a"] + 2 * holdout_data["column_b"] - 1.1 * holdout_data["column_d"] + 0.5
    )

    return real_data, synthetic_data, holdout_data


def get_classification_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    real_data = pd.DataFrame(
        {
            "column_a": 1.2 * np.random.randn(2500) + 1,
            "column_b": 2.5 * np.random.randn(2500) - 1,
            "column_c": choices(["cat", "horse", "dog"], weights=[0.25, 0.5, 0.25], k=2500),
            "column_d": 1.5 * np.random.randn(2500) + 12,
            "column_e": choices([0, 1], weights=[0.5, 0.5], k=2500),
        }
    )
    synthetic_data = pd.DataFrame(
        {
            "column_a": 1.1 * np.random.randn(2500) + 0.5,
            "column_b": 2.2 * np.random.randn(2500) - 1,
            "column_c": choices(["cat", "horse", "dog"], weights=[0.35, 0.35, 0.3], k=2500),
            "column_d": 1.5 * np.random.randn(2500) + 10,
            "column_e": choices([0, 1], weights=[0.25, 0.75], k=2500),
        }
    )
    return real_data, synthetic_data


def test_multi_target_modeling_difference_single_regression_target() -> None:
    # This should be the same as if we used the MeanRegressionDifference class directly.
    set_all_random_seeds(42)

    real_data, synthetic_data, holdout_data = get_regression_data()

    metric = MultiTargetModelingDifference(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b", "column_d", "column_e"],
        do_preprocess=True,
        preprocess_labels=False,
        label_columns_and_type={"column_e": ColumnType.NUMERICAL},
        regressors_config_path=Path("tests/assets/regression_config_1.json"),
    )

    with pytest.raises(AssertionError):
        # Should fail without holdout
        score = metric.compute(real_data, synthetic_data)

    score = metric.compute(real_data, synthetic_data, holdout_data)

    assert pytest.approx(-0.03548026558881601, abs=1e-8) == score["avg_r2_difference"]
    assert pytest.approx(1.0157588258591808, abs=1e-8) == score["avg_mean_squared_error_difference"]

    # Since we're just regressing on one label, the combined regression and f1 metrics should stay the same
    assert pytest.approx(-0.03548026558881601, abs=1e-8) == score["avg_r2_difference_and_f1_difference"]
    assert pytest.approx(1.0157588258591808, abs=1e-8) == score["avg_mean_squared_error_difference_and_f1_difference"]

    unset_all_random_seeds()


def test_multi_target_modeling_difference_single_regression_target_with_original_labels() -> None:
    # This should be the same as if we used the MeanRegressionDifference class directly.
    set_all_random_seeds(42)

    real_data, synthetic_data, holdout_data = get_regression_data()
    # Scale the labels by 100
    real_data["column_e"] = 100 * real_data["column_e"]
    synthetic_data["column_e"] = 100 * synthetic_data["column_e"]
    holdout_data["column_e"] = 100 * holdout_data["column_e"]

    metric_1 = MultiTargetModelingDifference(
        categorical_columns=[],
        numerical_columns=["column_a", "column_b", "column_d", "column_e"],
        do_preprocess=True,
        preprocess_labels=True,
        label_columns_and_type={"column_e": ColumnType.NUMERICAL},
        regressors_config_path=Path("tests/assets/regression_config_2.json"),
        measure_metrics_in_original_label_space=True,
    )

    metric_2 = MultiTargetModelingDifference(
        categorical_columns=[],
        numerical_columns=["column_a", "column_b", "column_d", "column_e"],
        do_preprocess=True,
        preprocess_labels=True,
        label_columns_and_type={"column_e": ColumnType.NUMERICAL},
        regressors_config_path=Path("tests/assets/regression_config_2.json"),
        measure_metrics_in_original_label_space=False,
    )

    score_1 = metric_1.compute(real_data, synthetic_data, holdout_data)
    score_2 = metric_2.compute(real_data, synthetic_data, holdout_data)

    assert score_1["avg_mean_squared_error_difference"] > score_2["avg_mean_squared_error_difference"]
    assert score_1["avg_mean_absolute_error_difference"] > score_2["avg_mean_absolute_error_difference"]

    unset_all_random_seeds()


def test_multi_target_modeling_difference_with_two_targets() -> None:
    # This should be the same for each target as if we had done the classification assessment separately
    set_all_random_seeds(42)

    real_data, synthetic_data = get_classification_data()
    holdout_data = real_data.copy()

    metric = MultiTargetModelingDifference(
        categorical_columns=["column_c", "column_e"],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
        preprocess_labels=False,
        f1_type="macro",
        label_columns_and_type={"column_e": ColumnType.CATEGORICAL, "column_a": ColumnType.NUMERICAL},
        regressors_config_path=Path("tests/assets/regression_config_1.json"),
        include_regressor_specific_averages=True,
    )

    score = metric.compute(real_data, synthetic_data, holdout_data)

    assert pytest.approx(-0.20674835173603345, abs=1e-8) == score["avg_f1_difference"]
    assert pytest.approx(-0.3210234643327887, abs=1e-8) == score["avg_r2_difference"]
    # average of the f1 and r2 difference values for the two columns of interest.
    assert pytest.approx(-0.2638859080344111, abs=1e-8) == score["avg_r2_difference_and_f1_difference"]

    assert pytest.approx(-0.1808825958690205, abs=1e-8) == score["MLPRegressor_r2_difference_and_f1_difference"]

    unset_all_random_seeds()


def test_multi_target_modeling_difference_with_two_cat_targets() -> None:
    set_all_random_seeds(42)

    real_data, synthetic_data = get_classification_data()
    holdout_data = real_data.copy()

    metric = MultiTargetModelingDifference(
        categorical_columns=["column_c", "column_e"],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
        preprocess_labels=False,
        f1_type="macro",
        label_columns_and_type={"column_e": ColumnType.CATEGORICAL, "column_c": ColumnType.CATEGORICAL},
        regressors_config_path=Path("tests/assets/regression_config_1.json"),
    )

    score = metric.compute(real_data, synthetic_data, holdout_data)

    assert pytest.approx(-0.15399437057829385, abs=1e-8) == score["avg_f1_difference"]

    # Test that nothing breaks when we have no numerical columns

    metric = MultiTargetModelingDifference(
        categorical_columns=["column_c", "column_e"],
        numerical_columns=[],
        do_preprocess=True,
        preprocess_labels=False,
        f1_type="macro",
        label_columns_and_type={"column_e": ColumnType.CATEGORICAL, "column_c": ColumnType.CATEGORICAL},
        regressors_config_path=Path("tests/assets/regression_config_1.json"),
    )

    score = metric.compute(real_data, synthetic_data, holdout_data)

    assert pytest.approx(-0.060709770749946976, abs=1e-8) == score["avg_f1_difference"]

    unset_all_random_seeds()


def test_multi_target_modeling_difference_custom_regression() -> None:
    # Test that custom regressors are applied for columns that have one in the config
    set_all_random_seeds(42)

    real_data, synthetic_data, holdout_data = get_regression_data()

    metric = MultiTargetModelingDifference(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b", "column_d", "column_e"],
        do_preprocess=True,
        preprocess_labels=False,
        label_columns_and_type={"column_b": ColumnType.NUMERICAL},
        regressors_config_path=Path("tests/assets/regression_config_2.json"),
        include_regressor_specific_averages=True,
    )

    score = metric.compute(real_data, synthetic_data, holdout_data)

    assert pytest.approx(-0.03296014796620539, abs=1e-8) == score["LinearRegression_r2_difference"]

    # We don't want any RandomForestRegressor metrics which would otherwise be applied in the default setting
    assert "RandomForestRegressor_r2_difference" not in score

    unset_all_random_seeds()


def test_multi_target_modeling_difference_exceptions() -> None:
    # Raise error when label column isn't in either column type list
    with pytest.raises(ValueError):
        _ = MultiTargetModelingDifference(
            categorical_columns=["column_c"],
            numerical_columns=["column_a", "column_b", "column_d"],
            do_preprocess=True,
            preprocess_labels=False,
            f1_type="macro",
            label_columns_and_type={"column_e": ColumnType.CATEGORICAL, "column_c": ColumnType.CATEGORICAL},
            regressors_config_path=Path("tests/assets/regression_config_1.json"),
        )

    # Raise error when column type doesn't match
    with pytest.raises(AssertionError):
        _ = MultiTargetModelingDifference(
            categorical_columns=["column_c", "column_e"],
            numerical_columns=["column_a", "column_b", "column_d"],
            do_preprocess=True,
            preprocess_labels=False,
            f1_type="macro",
            label_columns_and_type={"column_e": ColumnType.NUMERICAL, "column_c": ColumnType.CATEGORICAL},
            regressors_config_path=Path("tests/assets/regression_config_1.json"),
        )


def test_multi_target_modeling_difference_with_two_parallel_targets() -> None:
    # This should function in exactly the same way as when you don't process everything in parallel. However, due
    # to the way randomness happens in parallel processing it isn't exactly the same.
    # NOTE: We can force randomness inside the threads by inserting pins inside the function, which has been done to
    # confirm this works properly
    set_all_random_seeds(42)

    real_data, synthetic_data = get_classification_data()
    holdout_data = real_data.copy()

    # This is required to address a tests hanging issue on linux machines. This forces MP to use spawning instead of
    # forking for all OSs. This test would hang if run as a full pytest suite but be fine run individually on linux.
    # See https://github.com/pytest-dev/pytest/issues/11174
    mp.set_start_method("spawn", force=True)

    metric = MultiTargetModelingDifference(
        categorical_columns=["column_c", "column_e"],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
        preprocess_labels=False,
        f1_type="macro",
        label_columns_and_type={"column_e": ColumnType.CATEGORICAL, "column_a": ColumnType.NUMERICAL},
        regressors_config_path=Path("tests/assets/regression_config_1.json"),
        include_regressor_specific_averages=False,
        n_jobs=2,
    )

    score = metric.compute(real_data, synthetic_data, holdout_data)

    assert pytest.approx(-0.20674835173603345, abs=1e-8) == score["avg_f1_difference"]
    assert pytest.approx(-0.3210234643327887, abs=1e-8) == score["avg_r2_difference"]
    # average of the f1 and r2 difference values for the two columns of interest.
    assert pytest.approx(-0.2638859080344111, abs=1e-8) == score["avg_r2_difference_and_f1_difference"]

    assert pytest.approx(0.12331561250109607, abs=1e-8) == score["avg_mean_squared_error_difference_and_f1_difference"]

    unset_all_random_seeds()
