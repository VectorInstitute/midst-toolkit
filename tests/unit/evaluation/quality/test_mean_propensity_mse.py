from random import choices

import numpy as np
import pandas as pd
import pytest

from midst_toolkit.common.random import set_all_random_seeds, unset_all_random_seeds
from midst_toolkit.evaluation.quality.mean_propensity_mse import MeanPropensityMeanSquaredError


def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    real_data = pd.DataFrame(
        {
            "column_a": 1.2 * np.random.randn(2500) + 1,
            "column_b": 2.5 * np.random.randn(2500) - 1,
            "column_c": choices(["cat", "horse", "dog"], weights=[0.25, 0.5, 0.25], k=2500),
            "column_d": 1.5 * np.random.randn(2500) + 12,
        }
    )
    synthetic_data = pd.DataFrame(
        {
            "column_a": 1.1 * np.random.randn(2500) + 0.5,
            "column_b": 2.2 * np.random.randn(2500) - 1,
            "column_c": choices(["cat", "horse", "dog"], weights=[0.35, 0.35, 0.3], k=2500),
            "column_d": 1.5 * np.random.randn(2500) + 10,
        }
    )
    return real_data, synthetic_data


def test_mean_propensity_mse_with_preprocess() -> None:
    set_all_random_seeds(42)

    real_data, synthetic_data = get_data()

    metric = MeanPropensityMeanSquaredError(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
    )

    score = metric.compute(real_data, synthetic_data)
    assert pytest.approx(0.08442776888821232, abs=1e-8) == score["avg_pmse"]
    assert pytest.approx(0.7566743656985974, abs=1e-8) == score["avg_macro_f1_score"]
    unset_all_random_seeds()


def test_mean_propensity_mse_with_no_categorical() -> None:
    set_all_random_seeds(42)

    real_data, synthetic_data = get_data()

    metric = MeanPropensityMeanSquaredError(
        categorical_columns=[],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
    )

    score = metric.compute(real_data, synthetic_data)
    assert pytest.approx(0.08000946858157684, abs=1e-8) == score["avg_pmse"]
    assert pytest.approx(0.7485073080661124, abs=1e-8) == score["avg_macro_f1_score"]
    unset_all_random_seeds()


def test_mean_propensity_mse_with_no_numerical_and_shortcut() -> None:
    set_all_random_seeds(42)

    real_data, synthetic_data = get_data()
    real_data["column_e"] = 1
    synthetic_data["column_e"] = 0

    metric = MeanPropensityMeanSquaredError(
        categorical_columns=["column_c", "column_e"],
        numerical_columns=[],
        do_preprocess=True,
    )

    score = metric.compute(real_data, synthetic_data)
    # pMSE should be close to 0.25 and F1 should be essentially 1 due to the shortcut.
    assert pytest.approx(0.24374514497992789, abs=1e-8) == score["avg_pmse"]
    assert pytest.approx(1.0, abs=1e-8) == score["avg_macro_f1_score"]
    unset_all_random_seeds()
