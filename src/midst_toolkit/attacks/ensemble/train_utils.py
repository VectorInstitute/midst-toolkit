"""Training utilities for ensemble attacks."""

import numpy as np
from sklearn.metrics import roc_curve


def get_tpr_at_fpr(
    true_membership: np.ndarray,
    predictions: np.ndarray,
    max_fpr: float = 0.1,
) -> float:
    """
    Calculates the True Positive Rate (TPR) at a specified False Positive Rate (FPR) threshold.

    Args:
        true_membership: Array of true binary labels (0 or 1).
        predictions: Array of predicted probabilities or scores.
        max_fpr: Maximum False Positive Rate threshold. Defaults to 0.1.

    Returns:
        TPR at the specified FPR threshold.
    """
    fpr, tpr, _ = roc_curve(true_membership, predictions)

    return max(tpr[fpr <= max_fpr])
