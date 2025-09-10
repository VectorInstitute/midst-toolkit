import pandas as pd
from syntheval.metrics.utility.metric_mutual_information import MutualInformation

from midst_toolkit.evaluation.metric_base import SynthEvalQualityMetric


class MutualInformationDifference(SynthEvalQualityMetric):
    def __init__(
        self,
        categorical_columns: list[str],
        numerical_columns: list[str],
        do_preprocess: bool = False,
        include_numerical_columns: bool = True,
    ):
        """
        This class computes the Froebenius norm of the difference between the Mutual Information (MI) score matrices
        associated with two dataframes being compared. The computation is based on the work below.

        Ping H, Stoyanovich J, Howe B. DataSynthesizer: privacy-preserving synthetic datasets. 2017
        Presented at: Proceedings of the 29th International Conference on Scientific and Statistical Database
        Management; 2017; Chicago. [doi:10.1145/3085504.3091117]

        It leverages ``normalized_mutual_info_score`` from sklearn under the hood. The function computes the MI
        matrices, comparing the individual columns of the dataframes to each other. Then the difference of the
        two matrices is taken and the Froebenius norm computed for the final score.

        NOTE: Mutual Information works well for categorical variables. However, by default, SynthEval essentially
        just converts numerical columns to string representations for the computation. This isn't a great idea for
        things like floats. By default, this class respects SynthEval's choice, but you can override it and compute
        MI difference score for categorical columns by setting ``include_numerical_columns`` to False, or providing an
        empty list for ``numerical_columns``.

        Args:
            categorical_columns: Column names corresponding to the categorical variables of any provided dataframe.
            numerical_columns: Column names corresponding to the numerical variables of any provided dataframe.
            do_preprocess: Whether or not to preprocess the dataframes with the default pipeline used by SynthEval.
                Defaults to False.
            include_numerical_columns: Whether to include any provided numerical columns in the MI difference score
                computation. See the note above for why you might not want to include them.
        """
        super().__init__(categorical_columns, numerical_columns, do_preprocess)
        self.include_numerical_columns = include_numerical_columns
        self.all_columns = categorical_columns + numerical_columns

    def compute(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict[str, float]:
        """
        Computes the Froebenius norm of the difference between the Mutual Information (MI) score matrices associated
        with  the ``real_data`` and ``synthetic_data`` dataframes. The computation is based on the work below.

        Ping H, Stoyanovich J, Howe B. DataSynthesizer: privacy-preserving synthetic datasets. 2017
        Presented at: Proceedings of the 29th International Conference on Scientific and Statistical Database
        Management; 2017; Chicago. [doi:10.1145/3085504.3091117]

        It leverages ``normalized_mutual_info_score`` from sklearn under the hood. The function computes the MI
        matrices, comparing the individual columns of the dataframes to each other. Then the difference of the
        two matrices is taken and the Froebenius norm computed for the final score.

        NOTE: Mutual Information works well for categorical variables. However, by default, SynthEval essentially
        just converts numerical columns to string representations for the computation. This isn't a great idea for
        things like floats. By default, this class respects SynthEval's choice, but you can override it and compute
        MI difference score for categorical columns by setting ``self.include_numerical_columns`` to False, or
        ``self.numerical_columns`` to an empty list.

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

        # NOTE: The SynthEval MutualInformation class ignores your column specifications by default. However, for
        # other classes (correlation_matrix_difference for example), specifying less than all of the columns restricts
        # the score computation to just those columns. To make this consistent we do that here, before passing to the
        # SynthEval class.
        filtered_real_data = (
            real_data[self.all_columns] if self.include_numerical_columns else real_data[self.categorical_columns]
        )
        filtered_synthetic_data = (
            synthetic_data[self.all_columns]
            if self.include_numerical_columns
            else synthetic_data[self.categorical_columns]
        )

        self.syntheval_metric = MutualInformation(
            real_data=filtered_real_data,
            synt_data=filtered_synthetic_data,
            hout_data=None,
            cat_cols=self.categorical_columns,
            num_cols=self.numerical_columns,
            do_preprocessing=False,
            verbose=False,
        )

        return self.syntheval_metric.evaluate()
