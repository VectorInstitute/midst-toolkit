import pandas as pd
import pytest

from midst_toolkit.evaluation.quality.mutual_information_difference import MutualInformationDifference


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


def test_correlation_matrix_diff_no_preprocess() -> None:
    metric = MutualInformationDifference(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=False,
    )

    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(3.5432854275916057, abs=1e-8) == score["mutual_inf_diff"]
    assert score["mi_mat_dims"] == 4


def test_correlation_matrix_diff_with_preprocess() -> None:
    metric = MutualInformationDifference(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
    )

    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(3.5432854275916057, abs=1e-8) == score["mutual_inf_diff"]
    assert score["mi_mat_dims"] == 4


def test_one_column_left_off() -> None:
    metric = MutualInformationDifference(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b"],
        do_preprocess=True,
    )

    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(2.506647565147822, abs=1e-8) == score["mutual_inf_diff"]
    assert score["mi_mat_dims"] == 3


def test_mixed_correlation_no_categoricals() -> None:
    # Also leaving off column_d
    metric = MutualInformationDifference(
        categorical_columns=[],
        numerical_columns=["column_a", "column_b"],
        do_preprocess=True,
    )

    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(2.3475591685761548, abs=1e-8) == score["mutual_inf_diff"]
    assert score["mi_mat_dims"] == 2


def test_mixed_correlation_no_numericals() -> None:
    metric = MutualInformationDifference(
        categorical_columns=["column_a", "column_c"],
        numerical_columns=[],
        do_preprocess=True,
    )

    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(0, abs=1e-8) == score["mutual_inf_diff"]
    assert score["mi_mat_dims"] == 2
