import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


DEFAULT_FPR_THRESHOLDS = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2]


def tpr_at_fpr(true_membership: np.ndarray, predicted_membership: np.ndarray, fpr_threshold: float = 0.1) -> float:
    """
    Provided a set of challenge points, the goal of membership inference is to classify whether a given point was part
    of the training set for the target model or not. This function calculates true positive rate (TPR) at a given
    false positive rate (FPR) threshold for membership inference.

    Args:
        true_membership: An array of values in {0,1} of shape (n_samples, ) indicating the membership of a challenge
            point. 0: "non-member", 1: "member".
        predicted_membership: An array of values in the range [0,1] of shape (n_samples, ) indicating the confidence
            that a challenge point is a member. The closer the value to 1, the more "confident" the predictor is about
            the hypothesis that the  challenge point is a member.
        fpr_threshold: This is the FPR at which we want to measure the TPR for predictions. Defaults to 0.1.

    Returns:
        The TPR at the provided FPR as estimated by the predictions against the labels.
    """
    assert len(true_membership) == len(predicted_membership), (
        "Membership predictions must be the same length as labels"
    )
    assert np.all(predicted_membership >= 0), "Some predictions are < 0"
    assert np.all(predicted_membership <= 1), "Some predictions are > 1"

    fpr, tpr, _ = roc_curve(true_membership, predicted_membership)
    fpr_measurements_below_threshold = fpr < fpr_threshold
    if np.any(fpr_measurements_below_threshold):
        # If any FPRs are below threshold, return best TPR seen.
        return max(tpr[fpr_measurements_below_threshold])
    return 0.0


def compute_mia_metrics(
    true_membership: np.ndarray, predicted_membership: np.ndarray, fpr_thresholds: list[float] | None = None
) -> dict[str, float | np.ndarray]:
    """
    Calculates MIA scores based on true membership and predictions. This includes true positive rate (TPR) at various
    fixed values of false positive rate (FPR), ROC/AUC, the best accuracy, and an MIA score (max(TPR - FPR) across
    various prediction thresholds).

    Args:
        true_membership: An array of values in {0,1} of shape (n_samples, ) indicating the membership of a challenge
            point. 0: "non-member", 1: "member".
        predicted_membership: An array of values in the range [0,1] of shape (n_samples, ) indicating the confidence
            that a challenge point is a member. The closer the value to 1, the more "confident" the predictor is about
            the hypothesis that the  challenge point is a member.
        fpr_thresholds: Set of FPR thresholds at which to measure TPR. If not provided, it defaults to a set list of
            predefined thresholds. Values should be between 0 and 1. Defaults to None.

    Returns:
        A dictionary with a suite of metrics quantifying the success of the membership inference across a number of
        dimensions.
    """
    mia_metrics: dict[str, float | np.ndarray] = {}
    fpr_thresholds = fpr_thresholds if fpr_thresholds else DEFAULT_FPR_THRESHOLDS
    for fpr_threshold in fpr_thresholds:
        # Just shifting 4 significant digits to create a string (0.0025 -> "0025")
        formatted_score_key = f"TPR_FPR_{int(1e4 * fpr_threshold)}"
        mia_metrics[formatted_score_key] = tpr_at_fpr(true_membership, predicted_membership, fpr_threshold)
    fpr, tpr, _ = roc_curve(true_membership, predicted_membership)
    mia_metrics["fpr"] = fpr
    mia_metrics["tpr"] = tpr
    mia_metrics["auc"] = roc_auc_score(true_membership, predicted_membership)
    mia_metrics["mia"] = np.max(tpr - fpr)
    mia_metrics["balanced_accuracy"] = np.max(((1 - fpr) + tpr) / 2)
    return mia_metrics
