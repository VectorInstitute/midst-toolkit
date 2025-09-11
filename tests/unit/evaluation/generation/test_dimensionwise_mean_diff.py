import pandas as pd
import pytest

from midst_toolkit.evaluation.generation.dimensionwise_mean_difference import DimensionwiseMeanDifference


def test_dimensionwise_mean_diff_no_preprocess() -> None:
    real = pd.DataFrame({"column_a": [1, 2, 3], "column_b": [4, 5, 6], "column_c": ["horse", "dog", "horse"]})
    synthetic = pd.DataFrame({"column_a": [1, 2, 3], "column_b": [4, 6, 6], "column_c": ["cat", "dog", "horse"]})
    metric = DimensionwiseMeanDifference(
        categorical_columns=["column_c"], numerical_columns=["column_a", "column_b"], do_preprocess=False
    )

    score = metric.compute(real, synthetic)
    assert pytest.approx(0.5 * (0.0 + 1 / 3), abs=1e-8) == score["dimensionwise_mean_difference"]


def test_dimensionwise_mean_diff_with_preprocess() -> None:
    real = pd.DataFrame({"column_a": [1, 2, 3], "column_b": [4, 5, 6], "column_c": ["horse", "dog", "horse"]})
    synthetic = pd.DataFrame({"column_a": [1, 2, 3], "column_b": [4, 6, 6], "column_c": ["cat", "dog", "horse"]})
    metric = DimensionwiseMeanDifference(
        categorical_columns=["column_c"], numerical_columns=["column_a", "column_b"], do_preprocess=True
    )

    score = metric.compute(real, synthetic)
    assert pytest.approx(0.5 * (0.0 + 0.5 / 3.0), abs=1e-8) == score["dimensionwise_mean_difference"]
