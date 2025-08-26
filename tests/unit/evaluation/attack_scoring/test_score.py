from pathlib import Path

import numpy as np
import pytest

from midst_toolkit.evaluation.attack_scoring.score import compute_mia_metrics, tpr_at_fpr
from midst_toolkit.evaluation.attack_scoring.score_html import generate_html


def test_tpr_at_fpr_function_bad_lengths() -> None:
    labels = np.random.randint(0, 2, size=5)
    predictions = np.random.rand(6)

    # Should throw because predictions is too long compared to labels.
    with pytest.raises(AssertionError) as _:
        tpr_at_fpr(labels, predictions)


def test_tpr_at_fpr_function_bad_ranges() -> None:
    np.random.seed(42)

    labels = np.random.randint(0, 2, size=5)
    predictions = np.random.rand(5) * 10

    # Should throw because predictions are in a bad range
    with pytest.raises(AssertionError) as _:
        tpr_at_fpr(labels, predictions)

    # Unset seed for safety
    np.random.seed()


def test_tpr_at_fpr_correct() -> None:
    np.random.seed(42)

    labels = np.random.randint(0, 2, size=300)
    predictions = np.random.rand(300)

    score = tpr_at_fpr(labels, predictions)
    assert pytest.approx(score, abs=1e-6) == 0.2080536912751678

    # Now with custom thresholds
    custom_fpr_threshold = 0.2
    score = tpr_at_fpr(labels, predictions, custom_fpr_threshold)
    assert pytest.approx(score, abs=1e-8) == 0.3087248322147651

    # Now with really small threshold
    custom_fpr_threshold = 1e-10
    score = tpr_at_fpr(labels, predictions, custom_fpr_threshold)
    assert pytest.approx(score, abs=1e-8) == 0.006711409395973154

    # Unset seed for safety
    np.random.seed()


def test_mia_scores() -> None:
    np.random.seed(42)

    labels = np.random.randint(0, 2, size=300)
    predictions = np.random.rand(300)

    mia_scores = compute_mia_metrics(labels, predictions, [1e-10, 0.1, 0.2])

    assert pytest.approx(mia_scores["TPR_FPR_0"], abs=1e-8) == 0.006711409395973154
    assert pytest.approx(mia_scores["TPR_FPR_1000"], abs=1e-8) == 0.2080536912751678
    assert pytest.approx(mia_scores["TPR_FPR_2000"], abs=1e-8) == 0.3087248322147651
    assert pytest.approx(mia_scores["mia"], abs=1e-6) == 0.14307303
    assert pytest.approx(mia_scores["balanced_accuracy"], abs=1e-8) == 0.5715365127338993


def test_html_construction(tmp_path: Path) -> None:
    np.random.seed(10)
    true_labels = np.random.randint(0, 2, size=1000)
    predictions = np.random.rand(1000)

    # Run scoring function
    scores = compute_mia_metrics(true_labels, predictions)

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

    # Unset random seed for safety
    np.random.seed()
