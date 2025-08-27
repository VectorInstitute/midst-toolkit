import numpy as np

from midst_toolkit.evaluation.attack_scoring.scoring import (
    AttackScore,
    Auc,
    BalancedAccuracy,
    MiaMetrics,
    MiaScore,
    TprAtFpr,
    TprFpr,
)


DEFAULT_FPR_THRESHOLDS = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2]
DEFAULT_METRICS_TO_COMPUTE = set(MiaMetrics)


class MembershipInferenceMetrics:
    def __init__(self, metrics_to_compute: set[MiaMetrics] | None = None, fpr_thresholds: list[float] | None = None):
        """
        Metrics class to compute all (or subset) of the metrics associated with membership inference attacks. This
        class allows for specification of which metrics you want to compute and to provide a set of false positive
        rate threshold, which are used in computing some of the metrics, such as TPR at FPR.

        Args:
            metrics_to_compute: An enum list that specifies which metrics to compute. If None, then all available
                metrics are scheduled to be computed. Defaults to None.
            fpr_thresholds: A set of thresholds at which to measure TPR. This only matters if computing TRP @ FPR
                metrics. If not specified a default set of thresholds is used defined by DEFAULT_FPR_THRESHOLDS.
                Defaults to None.
        """
        self.fpr_thresholds = fpr_thresholds if fpr_thresholds else DEFAULT_FPR_THRESHOLDS
        self.metrics_to_compute = metrics_to_compute if metrics_to_compute else DEFAULT_METRICS_TO_COMPUTE
        self.metrics = self._map_enum_to_metrics()
        self.computed = False

    def _map_enum_to_metrics(self) -> list[AttackScore]:
        """
        Converts the list of enums specifying which metrics to use into objects that will compute the metrics.

        Returns:
            A set of metrics objects to be used for computing the desired metrics.
        """
        metrics: list[AttackScore] = []
        if MiaMetrics.BALANCED_ACCURACY in self.metrics_to_compute:
            metrics.append(BalancedAccuracy())
        if MiaMetrics.AUC in self.metrics_to_compute:
            metrics.append(Auc())
        if MiaMetrics.MIA_SCORE in self.metrics_to_compute:
            metrics.append(MiaScore())
        if MiaMetrics.TPR_AT_FPR in self.metrics_to_compute:
            metrics.append(TprAtFpr(self.fpr_thresholds))

        tpr_fpr_intersection = self.metrics_to_compute.intersection({MiaMetrics.TPR, MiaMetrics.FPR})
        if len(tpr_fpr_intersection) > 0:
            metrics.append(TprFpr(tpr_fpr_intersection))
        return metrics

    def compute(self, true_membership: np.ndarray, predicted_membership: np.ndarray) -> None:
        """
        Calculates MIA scores based on true membership and predictions. The set of metrics computed depends on the
        user specified values in metrics_to_compute. Options include true positive rate (TPR) at various fixed values
        of false positive rate (FPR), ROC/AUC, the best accuracy, and an MIA score (max(TPR - FPR) across various
        prediction thresholds).

        Args:
            true_membership: An array of values in {0,1} of shape (n_samples, ) indicating the membership of a
                challenge point. 0: "non-member", 1: "member".
            predicted_membership: An array of values in the range [0,1] of shape (n_samples, ) indicating the
                confidence that a challenge point is a member. The closer the value to 1, the more "confident" the
                predictor is about the hypothesis that the  challenge point is a member.
        """
        self.computed = True
        for metric in self.metrics:
            metric.compute(true_membership, predicted_membership)

    def to_dict(self) -> dict[str, float | np.ndarray]:
        """
        Converts to computed metrics into a dictionary for processing or reporting.

        Returns:
            A dictionary keyed by the name of the metric and either a metric score or set of scores in the form of a
            numpy array.
        """
        assert self.computed, "Compute has not been run for this metrics class. This must be run first."
        mia_metrics: dict[str, float | np.ndarray] = {}
        for metric in self.metrics:
            metric_dict = metric.to_dict()
            mia_metrics.update(metric_dict)

        return mia_metrics
