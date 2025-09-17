from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


class MiaMetrics(Enum):
    TPR_AT_FPR = "tpr_at_fpr"
    AUC = "auc"
    MIA_SCORE = "mia"
    BALANCED_ACCURACY = "balanced_accuracy"
    FPR = "fpr"
    TPR = "tpr"


class AttackScore(ABC):
    @abstractmethod
    def compute(self, true_membership: np.ndarray, predicted_membership: np.ndarray) -> None:
        """
        Compute function converting the provided labels and prediction into a score or set of scores.

        Args:
            true_membership: An array of values in {0,1} of shape (n_samples, ) indicating the membership of a
                challenge point. 0: "non-member", 1: "member".
            predicted_membership: An array of values in the range [0,1] of shape (n_samples, ) indicating the
                confidence that a challenge point is a member. The closer the value to 1, the more "confident" the
                predictor is about the hypothesis that the  challenge point is a member.

        Raises:
            NotImplementedError: Must be implemented by inheriting classes.
        """
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> dict[str, float | np.ndarray]:
        """
        Converts the computed metrics into a dictionary for processing or reporting.

        Raises:
            NotImplementedError: To be implemented by inheriting class

        Returns:
            A dictionary keyed by the name of the metric and either a metric score or set of scores in the form of a
            numpy array.
        """
        raise NotImplementedError


class MiaScore(AttackScore):
    def __init__(self, score_key: str = MiaMetrics.MIA_SCORE.value):
        """
        Class for computing an MIA score. This score is the maximum difference between TPR and FPR across the
        TPR and FPR curves.

        Args:
            score_key: Metric name/key to use in the metric dictionary. Defaults to MiaMetrics.MIA_SCORE.value.
        """
        self.score_key = score_key
        self.score: float | None = None

    def compute(self, true_membership: np.ndarray, predicted_membership: np.ndarray) -> None:
        """
        Compute the MIA score. This score is the maximum difference between TPR and FPR across the TPR and FPR curves.

        Args:
            true_membership: An array of values in {0,1} of shape (n_samples, ) indicating the membership of a
                challenge point. 0: "non-member", 1: "member".
            predicted_membership: An array of values in the range [0,1] of shape (n_samples, ) indicating the
                confidence that a challenge point is a member. The closer the value to 1, the more "confident" the
                predictor is about the hypothesis that the  challenge point is a member.
        """
        fpr, tpr, _ = roc_curve(true_membership, predicted_membership)
        self.score = np.max(tpr - fpr)

    def to_dict(self) -> dict[str, float | np.ndarray]:
        """
        Converts the computed MIA score into a dictionary for processing or reporting.

        Returns:
            A dictionary keyed by the name of the metric and the computed MIA score
        """
        assert self.score is not None, "Score is None, compute may not have been called yet."
        return {self.score_key: self.score}


class Auc(AttackScore):
    def __init__(self, score_key: str = MiaMetrics.AUC.value):
        """
        Class for computing an AUC/ROC score.

        Args:
            score_key:  Metric name/key to use in the metric dictionary. Defaults to MiaMetrics.AUC.value.
        """
        self.score_key = score_key
        self.score: float | None = None

    def compute(self, true_membership: np.ndarray, predicted_membership: np.ndarray) -> None:
        """
        Compute the AUC/ROC score for the provided labels and predictions.

        Args:
            true_membership: An array of values in {0,1} of shape (n_samples, ) indicating the membership of a
                challenge point. 0: "non-member", 1: "member".
            predicted_membership: An array of values in the range [0,1] of shape (n_samples, ) indicating the
                confidence that a challenge point is a member. The closer the value to 1, the more "confident" the
                predictor is about the hypothesis that the  challenge point is a member.
        """
        self.score = roc_auc_score(true_membership, predicted_membership)

    def to_dict(self) -> dict[str, float | np.ndarray]:
        """
        Converts the computed AUC/ROC into a dictionary for processing or reporting.

        Returns:
            A dictionary keyed by the name of the metric and the computed AUC/ROC score
        """
        assert self.score is not None, "Score is None, compute may not have been called yet."
        return {self.score_key: self.score}


class BalancedAccuracy(AttackScore):
    def __init__(self, score_key: str = MiaMetrics.BALANCED_ACCURACY.value):
        """
        Class for computing an Balanced Accuracy score.

        Args:
            score_key:  Metric name/key to use in the metric dictionary. Defaults to
                MiaMetrics.BALANCED_ACCURACY.value.
        """
        self.score_key = score_key
        self.score: float | None = None

    def compute(self, true_membership: np.ndarray, predicted_membership: np.ndarray) -> None:
        """
        Compute the Balanced Accuracy score for the provided labels and predictions.

        Args:
            true_membership: An array of values in {0,1} of shape (n_samples, ) indicating the membership of a
                challenge point. 0: "non-member", 1: "member".
            predicted_membership: An array of values in the range [0,1] of shape (n_samples, ) indicating the
                confidence that a challenge point is a member. The closer the value to 1, the more "confident" the
                predictor is about the hypothesis that the  challenge point is a member.
        """
        fpr, tpr, _ = roc_curve(true_membership, predicted_membership)
        self.score = np.max(((1 - fpr) + tpr) / 2)

    def to_dict(self) -> dict[str, float | np.ndarray]:
        """
        Converts the computed balanced accuracy score into a dictionary for processing or reporting.

        Returns:
            A dictionary keyed by the name of the metric and the computed balanced accuracy score
        """
        assert self.score is not None, "Score is None, compute may not have been called yet."
        return {self.score_key: self.score}


class TprFpr(AttackScore):
    def __init__(
        self,
        config: set[MiaMetrics] | None = None,
        fpr_score_key: str = MiaMetrics.FPR.value,
        tpr_score_key: str = MiaMetrics.TPR.value,
    ):
        """
        Class for computing an the TPR and FPR curves score.

        Args:
            config: Whether to include one or both of FPR and TPR curves. If none, then both curves are computed and
                returned. Defaults to None.
            fpr_score_key:  Metric name/key to use in the metric dictionary for the FPR curves. Defaults to
                MiaMetrics.FPR.value.
            tpr_score_key:  Metric name/key to use in the metric dictionary for the TPR curves. Defaults to
                MiaMetrics.TPR.value.
        """
        self.config = config if config else {MiaMetrics.FPR, MiaMetrics.TPR}
        assert len(self.config) > 0, "Configuration is empty."

        self.fpr_score_key = fpr_score_key
        self.tpr_score_key = tpr_score_key
        self.fpr: np.ndarray | None = None
        self.tpr: np.ndarray | None = None

    def compute(self, true_membership: np.ndarray, predicted_membership: np.ndarray) -> None:
        """
        Compute the TPR and FPR curves for the provided labels and predictions.

        Args:
            true_membership: An array of values in {0,1} of shape (n_samples, ) indicating the membership of a
                challenge point. 0: "non-member", 1: "member".
            predicted_membership: An array of values in the range [0,1] of shape (n_samples, ) indicating the
                confidence that a challenge point is a member. The closer the value to 1, the more "confident" the
                predictor is about the hypothesis that the  challenge point is a member.
        """
        self.fpr, self.tpr, _ = roc_curve(true_membership, predicted_membership)

    def to_dict(self) -> dict[str, float | np.ndarray]:
        """
        Converts the computed TPR and FPR curves into a dictionary for processing or reporting.

        Returns:
            A dictionary keyed by the name of the TPR and FPR metrics and the numpy arrays describing the TPR and FPR
            curves.
        """
        assert self.fpr is not None, "FPR is None, compute may not have been called yet."
        assert self.tpr is not None, "TPR is None, compute may not have been called yet."
        results_dict: dict[str, float | np.ndarray] = {}
        if MiaMetrics.FPR in self.config:
            results_dict[self.fpr_score_key] = self.fpr
        if MiaMetrics.TPR in self.config:
            results_dict[self.tpr_score_key] = self.tpr
        return results_dict


class TprAtFpr(AttackScore):
    def __init__(self, fpr_thresholds: list[float]):
        """
        Class for computing an the TPR at FPR thresholds for the provided set of thresholds.

        Args:
            fpr_thresholds: These are the various thresholds for FPR at which the TPR values will be measured.
        """
        assert len(fpr_thresholds) > 0, "No FPR Thresholds specified, fpr_threshold list is empty"
        self.fpr_thresholds = fpr_thresholds
        self.tpr_at_fprs: dict[str, float | np.ndarray] = {}

    def compute(self, true_membership: np.ndarray, predicted_membership: np.ndarray) -> None:
        """
        Provided a set of challenge points, the goal of membership inference is to classify whether a given point was
        part of the training set for the target model or not. This function calculates true positive rate (TPR) at a
        given set of false positive rate (FPR) thresholds for membership inference.

        Args:
            true_membership: An array of values in {0,1} of shape (n_samples, ) indicating the membership of a
                challenge point. 0: "non-member", 1: "member".
            predicted_membership: An array of values in the range [0,1] of shape (n_samples, ) indicating the
                confidence that a challenge point is a member. The closer the value to 1, the more "confident" the
                predictor is about the hypothesis that the  challenge point is a member.
        """
        assert len(true_membership) == len(predicted_membership), (
            "Membership predictions must be the same length as labels"
        )
        assert np.all(predicted_membership >= 0), "Some predictions are < 0"
        assert np.all(predicted_membership <= 1), "Some predictions are > 1"

        fpr, tpr, _ = roc_curve(true_membership, predicted_membership)

        for fpr_threshold in self.fpr_thresholds:
            # Just shifting 4 significant digits to create a string (0.0025 -> "25")
            formatted_score_key = f"TPR_FPR_{int(1e4 * fpr_threshold)}"
            fpr_measurements_below_threshold = fpr < fpr_threshold
            if np.any(fpr_measurements_below_threshold):
                # If any FPRs are below threshold, return best TPR seen.
                self.tpr_at_fprs[formatted_score_key] = max(tpr[fpr_measurements_below_threshold])
            else:
                self.tpr_at_fprs[formatted_score_key] = 0.0

    def to_dict(self) -> dict[str, float | np.ndarray]:
        """
        Converts the computed TPR at FPR metrics at various thresholds into a dictionary for processing or reporting.

        Returns:
            A dictionary keyed by the name of the TPR at FPR for various threshold and the computed TPR values for
            that threshold
        """
        return self.tpr_at_fprs
