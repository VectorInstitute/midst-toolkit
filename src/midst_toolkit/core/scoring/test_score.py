import os

import numpy as np

from .score import score
from .score_html import generate_html


def test_random_scores(tmp_path: str = ".") -> None:
    """Test scoring with random data and generate HTML output."""
    # Generate random true labels and prediction scores
    np.random.seed(10)  # For reproducibility
    true_labels = np.random.randint(0, 2, size=1000)
    predictions = np.random.rand(1000)

    # Run scoring function
    scores = score(true_labels.tolist(), predictions.tolist())
    fpr = scores["fpr"]
    print("Min FPR:", fpr.min())
    print("FPR values below 0.01:", fpr[fpr < 0.01])

    # Check that required metrics are present
    required_keys = [
        "AUC",
        "MIA",
        "accuracy",
        "fpr",
        "tpr",
        "TPR_FPR_10",
        "TPR_FPR_100",
        "TPR_FPR_500",
        "TPR_FPR_1000",
        "TPR_FPR_1500",
        "TPR_FPR_2000",
    ]
    for key in required_keys:
        assert key in scores, f"Missing key: {key}"
        assert isinstance(scores[key], (float, np.ndarray)), f"Wrong type for: {key}"

    # Wrap scores in the expected scenario format
    all_scores = {"test_scenario": scores}

    # Generate HTML
    html = generate_html(all_scores)

    # Optionally write to disk for manual inspection
    if tmp_path:
        with open(os.path.join(tmp_path, "test_scores.html"), "w") as f:
            f.write(html)

    print("Scoring test passed. HTML generated.")


if __name__ == "__main__":
    test_random_scores()
