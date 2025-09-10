from enum import IntEnum

import pandas as pd
from syntheval.metrics.utility.metric_confidence_interval_overlap import ConfidenceIntervalOverlap

from midst_toolkit.evaluation.metric_base import SynthEvalQualityMetric


class ConfidenceLevel(IntEnum):
    Eighty = 80
    Ninety = 90
    NinetyFive = 95
    NinetyEight = 98
    NinetyNine = 99


class MeanConfidenceInternalOverlap(SynthEvalQualityMetric):
    def __init__(
        self,
        categorical_columns: list[str],
        numerical_columns: list[str],
        do_preprocess: bool = False,
        confidence_level: ConfidenceLevel = ConfidenceLevel.NinetyFive,
    ) -> None:
        """
        This class computes the mean of the interval overlap percentages for the confidence intervals (CIs) of each
        NUMERICAL column. The confidence intervals are interval estimates for the mean value of a particular column
        Within each column the value is the average percentage of overlap for the real and synthetic column CIs.

        For example:

        If the real column has CI [1.0, 3.0] and the synthetic column has CI [2.0, 5.0], then the overlap width is 1.0
        and the overlap value for the column is 0.5*(1.0/2.0 + 1.0/3.0), where 2.0 is the width of the first interval
        and 3.0 is the width of the second. The final score is the average of these computations across all columns.

        This metric also computes the number of intervals that DO NOT overlap, the percentage thereof, and an estimate
        on the uncertainty of the mean number of overlaps themselves.

        NOTE: This uses z-scores for confidence interval construction NOT t-scores

        Args:
            categorical_columns: Column names corresponding to the categorical variables of any provided dataframe.
            numerical_columns: Column names corresponding to the numerical variables of any provided dataframe.
            do_preprocess: Whether or not to preprocess the dataframes with the default pipeline used by SynthEval.
                Defaults to False.
            confidence_level: The confidence level for confidence interval calculations for each of the numerical
                columns. These are the confidence intervals constructed around the mean of the numerical column
                values. Defaults to ConfidenceLevel.NinetyFive.
        """
        super().__init__(categorical_columns, numerical_columns, do_preprocess)
        self.confidence_level = confidence_level

    def compute(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict[str, float]:
        """
        Computes the mean of the interval overlap percentages for the confidence intervals (CIs) of each
        NUMERICAL column. The confidence intervals are interval estimates for the mean value of a particular column
        Within each column the value is the average percentage of overlap for the real and synthetic column CIs.

        For example:

        If the real column has CI [1.0, 3.0] and the synthetic column has CI [2.0, 5.0], then the overlap width is 1.0
        and the overlap value for the column is 0.5*(1.0/2.0 + 1.0/3.0), where 2.0 is the width of the first interval
        and 3.0 is the width of the second. The final score is the average of these computations across all columns.

        This metric also computes the number of intervals that DO NOT overlap, the percentage thereof, and an estimate
        on the uncertainty of the mean number of overlaps themselves.

        Args:
            real_data: Real data to which the synthetic data may be compared. In many cases this will be data used
                to TRAIN the model that generated the synthetic data, but not always.
            synthetic_data: Synthetically generated data whose quality is to be assessed.

        Returns:
            A dictionary with the various metric values for confidence interval overlap.

            - "avg overlap": The mean of the overlap value for all numerical columns.
            - "overlap err": An estimate of the uncertainty associated with the overlap percentage.
            - "num non-overlaps": This is the number of columns whose confidence interval DO NOT overlap.
            - "frac non-overlaps": It's the percentage of columns without a CI overlap.
        """
        if self.do_preprocess:
            real_data, synthetic_data, _ = self.preprocess(real_data, synthetic_data)

        self.syntheval_metric = ConfidenceIntervalOverlap(
            real_data=real_data,
            synt_data=synthetic_data,
            hout_data=None,
            cat_cols=self.categorical_columns,
            num_cols=self.numerical_columns,
            do_preprocessing=False,
            verbose=False,
        )

        return self.syntheval_metric.evaluate(self.confidence_level.value)
