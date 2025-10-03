import pandas as pd
import pytest

from midst_toolkit.evaluation.privacy.hitting_rate import HittingRate


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


def test_hitting_rate_with_preprocess() -> None:
    metric = HittingRate(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
    )
    # Should be 1/5 where the hit comes from the third row of the two dataframes.
    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(1 / 5, abs=1e-8) == score["hitting_rate"]


def test_one_column_left_off() -> None:
    metric = HittingRate(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_d"],
        do_preprocess=True,
    )

    # We're leaving off the column that has the most differences (column_b) so our hit rate should bump up to 4/5
    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(4 / 5, abs=1e-8) == score["hitting_rate"]


def test_hitting_rate_no_categoricals() -> None:
    # Also leaving off column_d
    metric = HittingRate(
        categorical_columns=[],
        numerical_columns=["column_a", "column_b"],
        do_preprocess=True,
    )

    # Hits should be the first and third rows of the dataframes
    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(2 / 5, abs=1e-8) == score["hitting_rate"]


def test_hitting_rate_no_numericals() -> None:
    metric = HittingRate(
        categorical_columns=["column_a", "column_c"],
        numerical_columns=[],
        do_preprocess=True,
    )

    # Hits should come from the last 4 rows.
    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(4 / 5, abs=1e-8) == score["hitting_rate"]


def test_hitting_rate_with_higher_threshold() -> None:
    metric = HittingRate(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
        hitting_threshold=0.3,
    )
    # Threshold is high enough that we should get another hit on the second row (in addition to the third)
    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(2 / 5, abs=1e-8) == score["hitting_rate"]
