# Possible training utilities for ensemble attacks.

import numpy as np
from sklearn.metrics import roc_curve


def get_tpr_at_fpr(
    true_membership: np.ndarray,
    predictions: np.ndarray,
    max_fpr: float = 0.1,
) -> float:
    """Calculates the best True Positive Rate when the False Positive Rate is at most `max_fpr`.

    :param true_membership: an array of values in {0,1} indicating the membership of each
        data point. 0: "non-member", 1: "member".
    :param predictions: an array of values in the range [0,1] indicating the confidence
            that a data point is a member.
    :param max_fpr: threshold on the FPR.

    return: The TPR at `max_fpr` FPR.
    """
    fpr, tpr, _ = roc_curve(true_membership, predictions)

    return max(tpr[fpr <= max_fpr])
