import pandas as pd
from syntheval.metrics.utility.metric_dimensionwise_means import MetricClassName as SynthEvalDwm

from midst_toolkit.evaluation.generation_quality.quality_metric_base import SynthEvalQualityMetric


class DimensionwiseMeanDifference(SynthEvalQualityMetric):
    def compute(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, holdout_data: pd.DataFrame | None = None
    ) -> dict[str, float]:
        """
        Function to compute the dimensionwise mean difference between the dataframe of real data and the provided
        synthetic data. The metric computes the mean value for each of the NUMERICAL columns in the two dataframes
        then computes the differences of these means per columns and then computes the mean of the absolute value
        of each of these differences. Ideally, this should be close to zero. An estimate of the cumulative standard
        error across all dimensions is also computed to approximate the variance associated with the cumulative
        difference of means across the numerical columns.

        Args:
            real_data: Real data to which the synthetic data may be compared. In many cases this will be data used
                to TRAIN the model that generated the synthetic data, but not always.
            synthetic_data: Synthetically generated data whose quality is to be assessed.
            holdout_data: This is UNUSED for this metric. Only the real and synthetic data are compared. Defaults to
                None.

        Returns:
            A dictionary with the cumulative difference of means between the real and synthetic data columns, keyed
            under "dimensionwise_mean_difference" and an estimate of the cumulative standard error across all
            dimensions. This is keyed under "standard_error_difference"
        """
        if self.do_preprocess:
            real_data, synthetic_data, holdout_data = self.preprocess(real_data, synthetic_data, holdout_data)

        self.syntheval_metric = SynthEvalDwm(
            real_data=real_data,
            synt_data=synthetic_data,
            cat_cols=self.categorical_columns,
            num_cols=self.numerical_columns,
            do_preprocessing=False,
            verbose=False,
        )

        result = self.syntheval_metric.evaluate()
        result["dimensionwise_mean_difference"] = result.pop("avg")
        result["standard_error_difference"] = result.pop("err")
        return result
