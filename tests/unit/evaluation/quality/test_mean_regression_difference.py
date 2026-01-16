from pathlib import Path
from random import choices

import numpy as np
import pandas as pd
import pytest

from midst_toolkit.common.random import set_all_random_seeds, unset_all_random_seeds
from midst_toolkit.evaluation.quality.mean_regression_difference import MeanRegressionDifference
from tests.utils.architecture import is_apple_silicon


def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
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


def test_mean_regression_diff_with_preprocess() -> None:
    set_all_random_seeds(42)

    real_data, synthetic_data, holdout_data = get_data()

    metric = MeanRegressionDifference(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
        preprocess_labels=False,
        label_column="column_e",
        include_additional_metrics=False,
        regressors_config=Path("tests/assets/regression_config_1.json"),
    )

    with pytest.raises(AssertionError):
        # Should fail without holdout
        score = metric.compute(real_data, synthetic_data)

    score = metric.compute(real_data, synthetic_data, holdout_data)

    # All negative, since the synthetic data isn't as good.
    assert pytest.approx(-0.028624301482850445, abs=1e-8) == score["LinearRegression_r2_difference"]
    assert pytest.approx(-0.003897949651049837, abs=1e-8) == score["avg_explained_variance_difference"]
    assert pytest.approx(-0.0006216447435568595, abs=1e-8) == score["MLPRegressor_explained_variance_difference"]

    # All  positive, since the synthetic data isn't as good.
    assert pytest.approx(0.8246922340902118, abs=1e-8) == score["LinearRegression_mean_squared_error_difference"]
    assert pytest.approx(0.812546726914048, abs=1e-8) == score["MLPRegressor_mean_squared_error_difference"]
    assert pytest.approx(0.8301723069599523, abs=1e-8) == score["avg_mean_absolute_error_difference"]
    assert pytest.approx(1.3503424674272537, abs=1e-8) == score["XGBRegressor_mean_squared_error_difference"]
    assert pytest.approx(0.75296750664711, abs=1e-8) == score["XGBRegressor_mean_absolute_error_difference"]

    assert pytest.approx(0.44689824206215023, abs=1e-8) == score["avg_all_scores_difference"]
    unset_all_random_seeds()


def test_mean_regression_diff_with_no_categorical() -> None:
    set_all_random_seeds(42)

    real_data, synthetic_data, holdout_data = get_data()

    metric = MeanRegressionDifference(
        categorical_columns=[],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
        preprocess_labels=True,
        include_additional_metrics=False,
        label_column="column_e",
        regressors_config=Path("tests/assets/regression_config_2.json"),
    )

    score = metric.compute(real_data, synthetic_data, holdout_data)
    # Due to numerical fluctuations on github runners, we have slightly different values.
    if is_apple_silicon():
        assert pytest.approx(-0.05650138480537015, abs=1e-8) == score["RandomForestRegressor_r2_difference"]
    else:
        assert pytest.approx(-0.05648892075668577, abs=1e-8) == score["RandomForestRegressor_r2_difference"]

    unset_all_random_seeds()


def test_mean_regression_diff_with_poor_synthetic() -> None:
    set_all_random_seeds(42)

    real_data, synthetic_data, holdout_data = get_data()
    # Replace with random numbers normally distributed around 10
    synthetic_data["column_e"] = np.random.randn(2500) + 10

    metric = MeanRegressionDifference(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
        preprocess_labels=True,
        label_column="column_e",
        include_additional_metrics=True,
        regressors_config=Path("tests/assets/regression_config_1.json"),
    )

    score = metric.compute(real_data, synthetic_data, holdout_data)

    # All very negative, since the synthetic data is bad.
    assert pytest.approx(-20.233048919780963, abs=1e-8) == score["LinearRegression_r2_difference"]
    assert pytest.approx(-1.0026103570042455, abs=1e-8) == score["avg_explained_variance_difference"]
    assert pytest.approx(-1.0161702020509602, abs=1e-8) == score["MLPRegressor_explained_variance_difference"]

    # All very positive, since the synthetic data is bad. (note the labels have been normalized so these should
    # be quite small if the synthetic data were good)
    assert pytest.approx(0.28731932765609774, abs=1e-8) == score["LinearRegression_mean_squared_error_difference"]
    assert pytest.approx(0.2849491598736113, abs=1e-8) == score["MLPRegressor_mean_squared_error_difference"]
    assert pytest.approx(0.5187444834458115, abs=1e-8) == score["avg_mean_absolute_error_difference"]
    assert pytest.approx(0.2868712032359326, abs=1e-8) == score["XGBRegressor_mean_squared_error_difference"]
    assert pytest.approx(0.5148243084549904, abs=1e-8) == score["XGBRegressor_mean_absolute_error_difference"]

    # Make sure we're getting the verbose stuff.
    assert pytest.approx(-19.35472471634642, abs=1e-8) == score["synthetic_avg_r2"]
    assert pytest.approx(0.00001933922579472451, abs=1e-8) == score["real_MLPRegressor_mean_squared_error"]

    unset_all_random_seeds()


def test_mean_regression_diff_with_original_labels() -> None:
    set_all_random_seeds(42)

    real_data, synthetic_data, holdout_data = get_data()
    # Scale the labels by 100
    real_data["column_e"] = 100 * real_data["column_e"]
    synthetic_data["column_e"] = 100 * synthetic_data["column_e"]
    holdout_data["column_e"] = 100 * holdout_data["column_e"]

    metric_1 = MeanRegressionDifference(
        categorical_columns=[],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
        preprocess_labels=True,
        include_additional_metrics=False,
        label_column="column_e",
        regressors_config=Path("tests/assets/regression_config_2.json"),
        measure_metrics_in_original_label_space=True,
    )

    metric_2 = MeanRegressionDifference(
        categorical_columns=[],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
        preprocess_labels=True,
        include_additional_metrics=False,
        label_column="column_e",
        regressors_config=Path("tests/assets/regression_config_2.json"),
        measure_metrics_in_original_label_space=False,
    )

    score_1 = metric_1.compute(real_data, synthetic_data, holdout_data)
    score_2 = metric_2.compute(real_data, synthetic_data, holdout_data)

    assert (
        score_1["RandomForestRegressor_mean_absolute_error_difference"]
        > score_2["RandomForestRegressor_mean_absolute_error_difference"]
    )
    unset_all_random_seeds()
