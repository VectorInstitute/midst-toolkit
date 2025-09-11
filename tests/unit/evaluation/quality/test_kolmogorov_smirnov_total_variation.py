import math

import pandas as pd
import pytest

from midst_toolkit.evaluation.quality.kolmogorov_smirnov_total_variation import KolmogorovSmirnovAndTotalVariation


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
        "column_c": ["cat", "dog", "horse", "cat", "river"],
        "column_d": [10, 11, 12, 34, 1],
    }
)


def test_ks_tvd_no_preprocess() -> None:
    metric = KolmogorovSmirnovAndTotalVariation(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=False,
    )

    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(1.6 / 4.0, abs=1e-8) == score["avg stat"]
    assert pytest.approx(1.4 / 3.0, abs=1e-8) == score["avg ks"]
    assert pytest.approx(0.2, abs=1e-8) == score["avg tvd"]
    assert pytest.approx((1.0 + 0.873015873015873 + 1.0 + 0.007936507936507936) / 4.0, abs=1e-8) == score["avg pval"]
    assert score["num sigs"] == 1
    assert score["frac sigs"] == 1 / 4.0


def test_ks_tvd_with_preprocess() -> None:
    metric = KolmogorovSmirnovAndTotalVariation(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
    )
    # Should be the same, as preprocessing just shifts and scales the distributions.
    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(1.6 / 4.0, abs=1e-8) == score["avg stat"]
    assert pytest.approx(1.4 / 3.0, abs=1e-8) == score["avg ks"]
    assert pytest.approx(0.2, abs=1e-8) == score["avg tvd"]
    assert pytest.approx((1.0 + 0.873015873015873 + 1.0 + 0.007936507936507936) / 4.0, abs=1e-8) == score["avg pval"]
    assert score["num sigs"] == 1
    assert score["frac sigs"] == 1 / 4.0


def test_one_column_left_off() -> None:
    metric = KolmogorovSmirnovAndTotalVariation(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b"],
        do_preprocess=True,
    )

    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(0.6 / 3.0, abs=1e-8) == score["avg stat"]
    assert pytest.approx(0.4 / 2.0, abs=1e-8) == score["avg ks"]
    assert pytest.approx(0.2, abs=1e-8) == score["avg tvd"]
    assert pytest.approx((1.0 + 0.873015873015873 + 1.0) / 3.0, abs=1e-8) == score["avg pval"]
    assert score["num sigs"] == 0
    assert score["frac sigs"] == 0.0


def test_ks_tvd_no_categoricals() -> None:
    # Also leaving off column_d
    metric = KolmogorovSmirnovAndTotalVariation(
        categorical_columns=[],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
    )

    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(1.4 / 3.0, abs=1e-8) == score["avg stat"]
    assert pytest.approx(1.4 / 3.0, abs=1e-8) == score["avg ks"]
    assert math.isnan(score["avg tvd"])
    assert pytest.approx((1.0 + 0.873015873015873 + 0.007936507936507936) / 3.0, abs=1e-8) == score["avg pval"]
    assert score["num sigs"] == 1
    assert score["frac sigs"] == 1 / 3.0


def test_ks_tvd_no_numericals() -> None:
    metric = KolmogorovSmirnovAndTotalVariation(
        categorical_columns=["column_b", "column_c"],
        numerical_columns=[],
        do_preprocess=True,
    )

    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(0.8 / 2.0, abs=1e-8) == score["avg stat"]
    assert math.isnan(score["avg ks"])
    assert pytest.approx(0.8 / 2.0, abs=1e-8) == score["avg tvd"]
    assert pytest.approx((1.0 + 1.0) / 2.0, abs=1e-8) == score["avg pval"]
    assert score["num sigs"] == 0
    assert score["frac sigs"] == 0.0


def test_ks_tvd_with_preprocess_new_confidence() -> None:
    metric = KolmogorovSmirnovAndTotalVariation(
        categorical_columns=["column_c"],
        numerical_columns=["column_a", "column_b", "column_d"],
        do_preprocess=True,
        significance_level=0.9,
    )
    # Stats shouldn't change, but the high significance level means the 0.873015873015873 should become significant.
    score = metric.compute(REAL_DATA, SYNTHETIC_DATA)
    assert pytest.approx(1.6 / 4.0, abs=1e-8) == score["avg stat"]
    assert pytest.approx(1.4 / 3.0, abs=1e-8) == score["avg ks"]
    assert pytest.approx(0.2, abs=1e-8) == score["avg tvd"]
    assert pytest.approx((1.0 + 0.873015873015873 + 1.0 + 0.007936507936507936) / 4.0, abs=1e-8) == score["avg pval"]
    assert score["num sigs"] == 2
    assert score["frac sigs"] == 2 / 4.0
