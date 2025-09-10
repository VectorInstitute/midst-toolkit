import pandas as pd
from syntheval.metrics.utility.metric_mutual_information import MutualInformation

from midst_toolkit.evaluation.metric_base import SynthEvalQualityMetric


class MutualInformationDifference(SynthEvalQualityMetric):
    def compute(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict[str, float]:
        """
        Computes the Froebenius norm of the difference between the Mutual Information (MI) score matrices associated
        with  the ``real_data`` and ``synthetic_data`` dataframes. The computation is based on.

        Ping H, Stoyanovich J, Howe B. DataSynthesizer: privacy-preserving synthetic datasets. 2017
        Presented at: Proceedings of the 29th International Conference on Scientific and Statistical Database
        Management; 2017; Chicago. [doi:10.1145/3085504.3091117]

        and leverages ``normalized_mutual_info_score`` from sklearn under the hood. The function computes the MI
        matrices, comparing the individual columns of the dataframes to each other. Then the difference of the
        two matrices is taken and the Froebenius norm computed for the final score.

        Args:
            real_data: Real data to which the synthetic data may be compared. In many cases this will be data used
                to TRAIN the model that generated the synthetic data, but not always.
            synthetic_data: Synthetically generated data whose quality is to be assessed.

        Returns:
            The Froebenius norm of the difference between the two real and synthetic data MI matrices and the
            number of columns in the computed mutual information (rows/columns count of the correlation matrices).
            These are keyed under 'mutual_inf_diff' and 'mi_mat_dims' respectively.
        """
        if self.do_preprocess:
            real_data, synthetic_data = self.preprocess(real_data, synthetic_data)

        self.syntheval_metric = MutualInformation(
            real_data=real_data,
            synt_data=synthetic_data,
            hout_data=None,
            cat_cols=self.categorical_columns,
            num_cols=self.numerical_columns,
            do_preprocessing=False,
            verbose=False,
        )

        return self.syntheval_metric.evaluate()
