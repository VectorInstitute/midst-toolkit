import math

import numpy as np
import pandas as pd
import pytest

from midst_toolkit.evaluation.quality.mean_hellinger_distance import MeanHellingerDistance


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
        "column_d": [-1, -2, -3, -2, -50],
    }
)

REAL_DATA_ENCODED = pd.DataFrame({"column_c": [1, 2, 1, 3, 3]})

SYNTHETIC_DATA_ENCODED = pd.DataFrame({"column_c": [3, 2, 1, 4, 3]})


def test_mean_hellinger_distance_with_no_preprocess() -> None:
    metric = MeanHellingerDistance(
        categorical_columns=["column_c"],
        numerical_columns=[],
        do_preprocess=False,
    )

    discrete_real = np.array([2 / 5, 1 / 5, 2 / 5])
    # 4 gets collapsed into the last bin
    synthetic_real = np.array([1 / 5, 1 / 5, 3 / 5])
    target = (1.0 / math.sqrt(2)) * np.linalg.norm(np.sqrt(discrete_real) - np.sqrt(synthetic_real))
    score = metric.compute(REAL_DATA_ENCODED, SYNTHETIC_DATA_ENCODED)
    assert pytest.approx(target, abs=1e-8) == score["mean_hellinger_distance"]
    assert np.isnan(score["hellinger_standard_error"])


def test_mean_hellinger_distance_with_preprocess_categorical() -> None:
    metric = MeanHellingerDistance(
        categorical_columns=["column_c"],
        numerical_columns=[],
        do_preprocess=True,
    )

    # Should be the same as after test_mean_hellinger_distance_with_no_preprocess running preprocessing
    discrete_real = np.array([2 / 5, 1 / 5, 2 / 5])
    # 4 gets collapsed into the last bin
    synthetic_real = np.array([1 / 5, 1 / 5, 3 / 5])
    target = (1.0 / math.sqrt(2)) * np.linalg.norm(np.sqrt(discrete_real) - np.sqrt(synthetic_real))
    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(target, abs=1e-8) == score["mean_hellinger_distance"]
    assert np.isnan(score["hellinger_standard_error"])


def test_mean_hellinger_distance_with_preprocess() -> None:
    metric = MeanHellingerDistance(
        categorical_columns=[],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
    )
    # Should be the same, as preprocessing doesn't change the categorical MI
    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(0.3598897091778779, abs=1e-8) == score["mean_hellinger_distance"]
    assert pytest.approx(0.18772239774180174, abs=1e-8) == score["hellinger_standard_error"]


def test_one_column_left_off() -> None:
    metric = MeanHellingerDistance(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b"],
        do_preprocess=True,
    )

    # Make sure computation doesn't include the column that was not included.
    target = 1 / 3 * (0.16510402468972515 + 0.0 + 0.6324555320336758)
    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(target, abs=1e-8) == score["mean_hellinger_distance"]


def test_mean_hellinger_distance_no_numericals() -> None:
    metric = MeanHellingerDistance(
        categorical_columns=["column_b", "column_c"],
        numerical_columns=[],
        do_preprocess=True,
    )

    # Everything should still work with an empty numerical list
    target = 1 / 2 * (0.3422824674525135 + 0.16510402468972515)
    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(target, abs=1e-8) == score["mean_hellinger_distance"]


def test_mean_hellinger_distance_do_not_include_numericals() -> None:
    metric = MeanHellingerDistance(
        categorical_columns=["column_b", "column_c"],
        numerical_columns=["column_a", "column_d"],
        do_preprocess=True,
        include_numerical_columns=False,
    )

    # Should be the same as test_mean_hellinger_distance_no_numericals since we're saying we do not want to include
    # numerical columns in the computations.
    target = 1 / 2 * (0.3422824674525135 + 0.16510402468972515)
    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(target, abs=1e-8) == score["mean_hellinger_distance"]
