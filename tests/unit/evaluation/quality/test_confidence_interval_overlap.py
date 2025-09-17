import pandas as pd
import pytest

from midst_toolkit.evaluation.quality.confidence_interval_overlap import (
    ConfidenceLevel,
    MeanConfidenceInternalOverlap,
)


def test_confidence_interval_overlap_no_preprocess() -> None:
    real = pd.DataFrame(
        {"column_a": [1, 2, 3], "column_b": [4, 5, 6], "column_c": ["horse", "dog", "horse"], "column_d": [1, 1, 2]}
    )
    synthetic = pd.DataFrame(
        {"column_a": [1, 2, 3], "column_b": [4, 6, 6], "column_c": ["cat", "dog", "horse"], "column_d": [6, 6, 7]}
    )

    # Default confidence level
    metric = MeanConfidenceInternalOverlap(
        categorical_columns=["column_c"], numerical_columns=["column_a", "column_b", "column_d"], do_preprocess=False
    )

    score = metric.compute(real, synthetic)

    # perfect overlap
    column_a_overlap = 1.0
    # disjoint
    column_d_overlap = 0.0

    column_b_overlap_width = 6.131606527611667 - 4.026666666666666
    column_b_real_ci_width = 6.131606527611667 - 3.868393472388333
    column_b_synthetic_ci_width = 6.64 - 4.026666666666666
    column_b_overlap = 0.5 * (
        column_b_overlap_width / column_b_real_ci_width + column_b_overlap_width / column_b_synthetic_ci_width
    )

    assert (
        pytest.approx(1.0 / 3.0 * (column_a_overlap + column_b_overlap + column_d_overlap), abs=1e-8)
        == score["avg overlap"]
    )
    assert score["num non-overlaps"] == 1.0
    assert score["frac non-overlaps"] == 1.0 / 3.0

    # 99 confidence level
    metric = MeanConfidenceInternalOverlap(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=False,
        confidence_level=ConfidenceLevel.NinetyNine,
    )

    score = metric.compute(real, synthetic)

    # perfect overlap
    column_a_overlap = 1.0
    # disjoint
    column_d_overlap = 0.0

    column_b_overlap_width = 6.4895636945092345 - 3.613333333333333
    column_b_real_ci_width = 6.4895636945092345 - 3.5104363054907655
    column_b_synthetic_ci_width = 7.053333333333333 - 3.613333333333333
    column_b_overlap = 0.5 * (
        column_b_overlap_width / column_b_real_ci_width + column_b_overlap_width / column_b_synthetic_ci_width
    )

    assert (
        pytest.approx(1.0 / 3.0 * (column_a_overlap + column_b_overlap + column_d_overlap), abs=1e-8)
        == score["avg overlap"]
    )
    assert score["num non-overlaps"] == 1.0
    assert score["frac non-overlaps"] == 1.0 / 3.0


def test_confidence_interval_overlap_with_preprocess() -> None:
    real = pd.DataFrame(
        {"column_a": [1, 2, 3], "column_b": [4, 5, 6], "column_c": ["horse", "dog", "horse"], "column_d": [1, 1, 2]}
    )
    synthetic = pd.DataFrame(
        {"column_a": [1, 2, 3], "column_b": [4, 6, 6], "column_c": ["cat", "dog", "horse"], "column_d": [6, 6, 7]}
    )

    metric = MeanConfidenceInternalOverlap(
        categorical_columns=["column_c"], numerical_columns=["column_a", "column_b", "column_d"], do_preprocess=True
    )

    score = metric.compute(real, synthetic)

    # perfect overlap
    column_a_overlap = 1.0
    # disjoint
    column_d_overlap = 0.0

    column_b_overlap_width = 1.0658032638058335 - 0.013333333333333197
    column_b_real_ci_width = 1.0658032638058335 - (-0.06580326380583335)
    column_b_synthetic_ci_width = 1.32 - 0.013333333333333197
    column_b_overlap = 0.5 * (
        column_b_overlap_width / column_b_real_ci_width + column_b_overlap_width / column_b_synthetic_ci_width
    )

    assert (
        pytest.approx(1.0 / 3.0 * (column_a_overlap + column_b_overlap + column_d_overlap), abs=1e-8)
        == score["avg overlap"]
    )
    assert score["num non-overlaps"] == 1.0
    assert score["frac non-overlaps"] == 1.0 / 3.0
