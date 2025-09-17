import pandas as pd
from syntheval.metrics.privacy.metric_hitting_rate import HittingRate as SynthEvalHittingRate

from midst_toolkit.evaluation.metrics_base import SynthEvalMetric


class HittingRate(SynthEvalMetric):
    def __init__(
        self,
        categorical_columns: list[str],
        numerical_columns: list[str],
        do_preprocess: bool = False,
        hitting_threshold: float = 1 / 30.0,
    ):
        """
        Class to determine the Hitting (Exact Match) rate associated with real and synthetic data. The rate is
        computed as the percentage of real data points provided that are "replicated" within the provided synthetic
        data.

        A synthetic data point is considered to "replicate" a real data point if each of its numerical values
        are within ``hitting_threshold`` percent of the variable range in the real data. For each categorical value an
        exact match is required.

        A smaller rate is better.

        NOTE: Categorical variables must be encoded in some way (ordinal or vector) for the evaluation to work. This
        can be accomplished by preprocessing the dataframes before calling compute or by setting ``do_preprocess`` to
        True.

        Args:
            categorical_columns: Column names corresponding to the categorical variables of any provided dataframe.
            numerical_columns: Column names corresponding to the numerical variables of any provided dataframe.
            do_preprocess: Whether or not to preprocess the dataframes with the default pipeline used by SynthEval.
                Defaults to False.
            hitting_threshold: For numerical columns, this determines the margin, in terms of percent of the total
                value range for a column an entry may vary by and still be considered a "hit." For example, if a
                column values ranges from 0 to 1 with a real value of 0.3 and synthetic value of 0.32, these values are
                still considered a match for a ``hitting_threshold`` of 1/30. Defaults to 1/30.
        """
        super().__init__(categorical_columns, numerical_columns, do_preprocess)
        assert 0 <= hitting_threshold <= 1.0, (
            f"Hitting threshold should be a value in [0, 1] but received: {hitting_threshold}"
        )
        self.hitting_threshold = hitting_threshold
        self.all_columns = categorical_columns + numerical_columns

    def compute(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict[str, float]:
        """
        Computes the Hitting (Exact Match) rate associated with real and synthetic data. The rate is computed as the
        percentage of real data points provided that are "replicated" within the provided synthetic  data.

        A synthetic data point is considered to "replicate" a real data point if each of its numerical values
        are within ``hitting_threshold`` percent of the variable range in the real data. For each categorical value an
        exact match is required.

        A smaller rate is better.

        NOTE: Categorical variables must be encoded in some way (ordinal or vector) for the evaluation to work. This
        can be accomplished by preprocessing the dataframes before calling compute or by setting ``do_preprocess`` to
        True.

        Args:
            real_data: Real data to which the synthetic data may be compared. In many cases this will be data used
                to TRAIN the model that generated the synthetic data, but not always.
            synthetic_data: Synthetically generated data whose quality is to be assessed.

        Returns:
            A dictionary containing the hitting rate, keyed by 'hitting_rate' a percentage of the real data points
            provided that are "replicated" in the synthetic data.
        """
        if self.do_preprocess:
            real_data, synthetic_data = self.preprocess(real_data, synthetic_data)

        # NOTE: The SynthEval HittingRate class ignores column specifications by default. However, for other classes
        # (correlation_matrix_difference for example), specifying less than all of the columns restricts the score
        # computation to just those columns. To make this consistent we do that here, before passing to the SynthEval
        # class.
        filtered_real_data = real_data[self.all_columns]
        filtered_synthetic_data = synthetic_data[self.all_columns]

        self.syntheval_metric = SynthEvalHittingRate(
            real_data=filtered_real_data,
            synt_data=filtered_synthetic_data,
            hout_data=None,
            cat_cols=self.categorical_columns,
            num_cols=self.numerical_columns,
            do_preprocessing=False,
            verbose=False,
        )
        result = self.syntheval_metric.evaluate(self.hitting_threshold)
        result["hitting_rate"] = result.pop("hit rate")
        return result
