import numpy as np
import pytest

from midst_toolkit.evaluation.attack_scoring.mia_metrics import DEFAULT_FPR_THRESHOLDS, MembershipInferenceMetrics
from midst_toolkit.evaluation.attack_scoring.score_html import generate_html
from midst_toolkit.evaluation.attack_scoring.scoring import MiaMetrics, TprAtFpr


DEFAULT_TPR_AT_FPR = TprAtFpr(DEFAULT_FPR_THRESHOLDS)


def test_tpr_at_fpr_function_bad_lengths() -> None:
    labels = np.random.randint(0, 2, size=5)
    predictions = np.random.rand(6)

    tpr_at_fpr = TprAtFpr([0.1])

    # Should throw because predictions is too long compared to labels.
    with pytest.raises(AssertionError) as _:
        tpr_at_fpr.compute(labels, predictions)


def test_tpr_at_fpr_function_bad_ranges() -> None:
    np.random.seed(42)

    labels = np.random.randint(0, 2, size=5)
    predictions = np.random.rand(5) * 10

    tpr_at_fpr = TprAtFpr([0.1])

    # Should throw because predictions are in a bad range
    with pytest.raises(AssertionError) as _:
        tpr_at_fpr.compute(labels, predictions)

    # Unset seed for safety
    np.random.seed()


def test_tpr_at_fpr_correct() -> None:
    np.random.seed(42)

    labels = np.random.randint(0, 2, size=300)
    predictions = np.random.rand(300)

    tpr_at_fpr = TprAtFpr([0.1])

    tpr_at_fpr.compute(labels, predictions)
    scores = tpr_at_fpr.to_dict()
    assert len(scores) == 1
    assert pytest.approx(scores["TPR_FPR_1000"], abs=1e-8) == 0.2080536912751678

    tpr_at_fpr = TprAtFpr([0.2])

    tpr_at_fpr.compute(labels, predictions)
    scores = tpr_at_fpr.to_dict()
    assert len(scores) == 1
    assert pytest.approx(scores["TPR_FPR_2000"], abs=1e-8) == 0.3087248322147651

    # Now with really small threshold

    tpr_at_fpr = TprAtFpr([1e-10])

    tpr_at_fpr.compute(labels, predictions)
    scores = tpr_at_fpr.to_dict()
    assert len(scores) == 1
    assert pytest.approx(scores["TPR_FPR_0"], abs=1e-8) == 0.006711409395973154

    # Unset seed for safety
    np.random.seed()


def test_mia_scores() -> None:
    np.random.seed(42)

    labels = np.random.randint(0, 2, size=300)
    predictions = np.random.rand(300)

    # Compute all metrics
    metric = MembershipInferenceMetrics(fpr_thresholds=[1e-10, 0.1, 0.2])

    metric.compute(labels, predictions)
    mia_scores = metric.to_dict()

    assert pytest.approx(mia_scores["TPR_FPR_0"], abs=1e-8) == 0.006711409395973154
    assert pytest.approx(mia_scores["TPR_FPR_1000"], abs=1e-8) == 0.2080536912751678
    assert pytest.approx(mia_scores["TPR_FPR_2000"], abs=1e-8) == 0.3087248322147651
    assert pytest.approx(mia_scores["mia"], abs=1e-6) == 0.14307303
    assert pytest.approx(mia_scores["balanced_accuracy"], abs=1e-8) == 0.5715365127338993

    # Test computing only a subset of metrics

    metric = MembershipInferenceMetrics(metrics_to_compute={MiaMetrics.TPR_AT_FPR, MiaMetrics.AUC, MiaMetrics.FPR})

    metric.compute(labels, predictions)
    mia_scores = metric.to_dict()

    assert "balanced_accuracy" not in mia_scores
    assert "mia" not in mia_scores
    assert "TPR_FPR_1000" in mia_scores
    assert "TPR_FPR_0" not in mia_scores
    assert "fpr" in mia_scores
    assert "tpr" not in mia_scores
    assert pytest.approx(mia_scores["TPR_FPR_1000"], abs=1e-8) == 0.2080536912751678


def test_html_construction() -> None:
    np.random.seed(10)
    true_labels = np.random.randint(0, 2, size=1000)
    predictions = np.random.rand(1000)

    # Default everything
    metric = MembershipInferenceMetrics()

    # Run scoring function
    metric.compute(true_labels, predictions)
    scores = metric.to_dict()

    # Check that required metrics are present in scores, because we asked for default values
    required_keys = {
        "auc",
        "mia",
        "balanced_accuracy",
        "fpr",
        "tpr",
        "TPR_FPR_10",
        "TPR_FPR_100",
        "TPR_FPR_500",
        "TPR_FPR_1000",
        "TPR_FPR_1500",
        "TPR_FPR_2000",
    }
    assert required_keys == set(scores.keys())

    # Wrap scores in the expected scenario format
    all_scores = {"test_scenario": scores}

    # Generate HTML
    html = generate_html(all_scores)
    assert len(html) > 0

    with open("/Users/david/Desktop/temp.html", "w") as f:
        f.write(html)

    # Unset random seed for safety
    np.random.seed()
