import pandas as pd
from syntheval.metrics.utility.metric_mixed_correlation import MixedCorrelation

from midst_toolkit.evaluation.metrics_base import SynthEvalMetric


class CorrelationMatrixDifference(SynthEvalMetric):
    def __init__(
        self,
        categorical_columns: list[str],
        numerical_columns: list[str],
        do_preprocess: bool = False,
        compute_mixed_correlations: bool = False,
    ):
        """
        This class computes the correlation matrices between each of the columns of both real and synthetic dataframes
        Then the difference between the correlation matrices is computed and the Froebenius norm of that difference
        is returned. A smaller norm is better.

        - Regardless of settings, correlations between the numerical columns are computed with Pearson correlation
          coefficients.
        - If ``compute_mixed_correlations`` is True, then correlations between the categorical variables is computed
          using Cramer's V, and correlations between the numerical and categorical variable is done with a correlation
          ratio, eta, as suggested in https://ieeexplore.ieee.org/document/10020639/.
        - If ``compute_mixed_correlations`` is False, ONLY numerical correlations are computed.

        NOTE: Categorical variables need not be one-hot encoded for correlations to work.

        Args:
            categorical_columns: Column names corresponding to the categorical variables of any provided dataframe.
            numerical_columns: Column names corresponding to the numerical variables of any provided dataframe.
            do_preprocess: Whether or not to preprocess the dataframes with the default pipeline used by SynthEval.
                Defaults to False.
            compute_mixed_correlations: Whether or not to compute correlations between the categorical variables and
                the categorical and numerical variables. See documentation above for a longer description. Defaults to
                False.
        """
        super().__init__(categorical_columns, numerical_columns, do_preprocess)
        self.compute_mixed_correlations = compute_mixed_correlations

    def compute(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict[str, float]:
        """
        Computes the Froebenius norm of the difference between the correlation matrices associated with the
        ``real_data`` and ``synthetic_data`` dataframes. The correlations computed depends on the value of
        ``compute_mixed_correlations`` for the class.

        - Regardless of settings, correlations between the numerical columns are computed with Pearson correlation
          coefficients.
        - If ``compute_mixed_correlations`` is True, then correlations between the categorical variables is computed
          using Cramer's V, and correlations between the numerical and categorical variable is done with a correlation
          ration, eta, as suggested in https://ieeexplore.ieee.org/document/10020639/.
        - If ``compute_mixed_correlations`` is False, ONLY numerical correlations are computed.

        Args:
            real_data: Real data to which the synthetic data may be compared. In many cases this will be data used
                to TRAIN the model that generated the synthetic data, but not always.
            synthetic_data: Synthetically generated data whose quality is to be assessed.

        Returns:
            The Froebenius norm of the difference between the two real and synthetic data correlation matrices and the
            number of columns in the computed correlations (rows/columns count of the correlation matrices). These
            are keyed under 'corr_mat_diff' and 'corr_mat_dims' respectively.
        """
        if self.do_preprocess:
            real_data, synthetic_data = self.preprocess(real_data, synthetic_data)

        self.syntheval_metric = MixedCorrelation(
            real_data=real_data,
            synt_data=synthetic_data,
            hout_data=None,
            cat_cols=self.categorical_columns,
            num_cols=self.numerical_columns,
            do_preprocessing=False,
            verbose=False,
        )

        return self.syntheval_metric.evaluate(mixed_corr=self.compute_mixed_correlations, return_mats=False)
