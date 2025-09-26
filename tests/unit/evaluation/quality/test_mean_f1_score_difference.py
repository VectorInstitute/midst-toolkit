from random import choices

import numpy as np
import pandas as pd
import pytest

from midst_toolkit.common.random import set_all_random_seeds, unset_all_random_seeds
from midst_toolkit.evaluation.quality.mean_f1_score_difference import MeanF1ScoreDifference


def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
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


def test_mean_f1_score_diff_with_preprocess() -> None:
    set_all_random_seeds(42)

    real_data, synthetic_data = get_data()

    metric = MeanF1ScoreDifference(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
        label_column="column_e",
    )

    score = metric.compute(real_data, synthetic_data)
    assert pytest.approx(0.7668, abs=1e-8) == score["random_forest_real_train_f1"]
    assert pytest.approx(0.5008, abs=1e-8) == score["random_forest_synthetic_train_f1"]
    assert pytest.approx(0.5, abs=1e-8) == score["adaboost_real_train_f1"]
    assert pytest.approx(0.49879999999999997, abs=1e-8) == score["adaboost_synthetic_train_f1"]
    assert pytest.approx(0.5064, abs=1e-8) == score["svm_real_train_f1"]
    assert pytest.approx(0.49960000000000004, abs=1e-8) == score["svm_synthetic_train_f1"]
    assert pytest.approx(0.49720000000000003, abs=1e-8) == score["logreg_real_train_f1"]
    assert pytest.approx(0.49960000000000004, abs=1e-8) == score["logreg_synthetic_train_f1"]
    assert pytest.approx(-0.06789999999999999, abs=1e-8) == score["mean_f1_difference"]
    unset_all_random_seeds()


def test_mean_f1_score_diff_with_no_categorical() -> None:
    set_all_random_seeds(42)

    real_data, synthetic_data = get_data()

    metric = MeanF1ScoreDifference(
        categorical_columns=[],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
        label_column="column_e",
    )

    score = metric.compute(real_data, synthetic_data)
    assert pytest.approx(-0.0792, abs=1e-8) == score["mean_f1_difference"]
    unset_all_random_seeds()


def test_mean_f1_score_diff_with_holdout_difference_f1() -> None:
    set_all_random_seeds(42)

    real_data, synthetic_data = get_data()
    holdout_data = real_data.copy()

    metric = MeanF1ScoreDifference(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
        label_column="column_e",
        f1_type="macro",
    )

    score = metric.compute(real_data, synthetic_data, holdout_data)
    assert pytest.approx(0.7667172638761903, abs=1e-8) == score["random_forest_real_train_f1"]
    assert pytest.approx(0.40831722022666145, abs=1e-8) == score["random_forest_synthetic_train_f1"]
    assert pytest.approx(0.3632940727026944, abs=1e-8) == score["adaboost_real_train_f1"]
    assert pytest.approx(0.33490261584802905, abs=1e-8) == score["adaboost_synthetic_train_f1"]
    assert pytest.approx(0.49933604817892385, abs=1e-8) == score["svm_real_train_f1"]
    assert pytest.approx(0.33315531820204713, abs=1e-8) == score["svm_synthetic_train_f1"]
    assert pytest.approx(0.4966708698534732, abs=1e-8) == score["logreg_real_train_f1"]
    assert pytest.approx(0.33315531820204713, abs=1e-8) == score["logreg_synthetic_train_f1"]
    assert pytest.approx(-0.17912194553312424, abs=1e-8) == score["mean_f1_difference"]

    # Spot check the holdout values.
    assert pytest.approx(-0.20674835173603345, abs=1e-8) == score["mean_f1_difference_holdout"]
    assert pytest.approx(0.8987963566688401, abs=1e-8) == score["random_forest_real_train_f1_holdout"]
    unset_all_random_seeds()
