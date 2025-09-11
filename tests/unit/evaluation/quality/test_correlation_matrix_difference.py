import pandas as pd
import pytest

from midst_toolkit.evaluation.quality.correlation_matrix_difference import CorrelationMatrixDifference


REAL_DATA = pd.DataFrame(
    {
        "column_a": [1, 2, 3, 4, 5],
        "column_b": [4, 5, 6, 7, 8],
        "column_c": ["horse", "dog", "horse", "cat", "cat"],
        "column_d": [-1, -2, -3, -4, -5],
    }
)
SYNTHETIC_DATA = pd.DataFrame(
    {
        "column_a": [1, 2, 3, 4, 5],
        "column_b": [4, 6, 6, -1, 1],
        "column_c": ["cat", "dog", "horse", "cat", "cat"],
        "column_d": [-1, -2, -3, -4, -5],
    }
)


def test_correlation_matrix_diff_no_preprocess() -> None:
    metric = CorrelationMatrixDifference(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=False,
        compute_mixed_correlations=True,
    )

    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(3.5432854275916057, abs=1e-8) == score["corr_mat_diff"]
    assert score["corr_mat_dims"] == 4


def test_correlation_matrix_diff_with_preprocess() -> None:
    metric = CorrelationMatrixDifference(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
        compute_mixed_correlations=True,
    )

    # Should be the same as when we don't pre-process, because correlation is invariant of shift and scale
    # operations.
    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(3.5432854275916057, abs=1e-8) == score["corr_mat_diff"]
    assert score["corr_mat_dims"] == 4


def test_correlation_matrix_diff_no_mixed_correlation() -> None:
    metric = CorrelationMatrixDifference(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
        compute_mixed_correlations=False,
    )

    # Only compute numeric-numeric column correlations.
    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(3.319950014673705, abs=1e-8) == score["corr_mat_diff"]
    assert score["corr_mat_dims"] == 3


def test_one_column_left_off() -> None:
    metric = CorrelationMatrixDifference(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b"],
        do_preprocess=True,
        compute_mixed_correlations=True,
    )

    # Only compute for the columns specified.
    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(2.506647565147822, abs=1e-8) == score["corr_mat_diff"]
    assert score["corr_mat_dims"] == 3


def test_mixed_correlation_no_categoricals() -> None:
    # Also leaving off column_d
    metric = CorrelationMatrixDifference(
        categorical_columns=[],
        numerical_columns=["column_a", "column_b"],
        do_preprocess=True,
        compute_mixed_correlations=True,
    )

    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(2.3475591685761548, abs=1e-8) == score["corr_mat_diff"]
    assert score["corr_mat_dims"] == 2

    # Make sure including column_d produces same score as test_correlation_matrix_diff_no_mixed_correlation
    metric = CorrelationMatrixDifference(
        categorical_columns=[],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
        compute_mixed_correlations=True,
    )

    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(3.319950014673705, abs=1e-8) == score["corr_mat_diff"]
    assert score["corr_mat_dims"] == 3


def test_mixed_correlation_no_numericals() -> None:
    metric = CorrelationMatrixDifference(
        categorical_columns=["column_a", "column_c"],
        numerical_columns=[],
        do_preprocess=True,
        compute_mixed_correlations=True,
    )

    # The columns are "perfectly" correlated, so we get a 0 difference.
    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(0, abs=1e-8) == score["corr_mat_diff"]
    assert score["corr_mat_dims"] == 2
