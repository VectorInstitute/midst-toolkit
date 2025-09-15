import pandas as pd
from syntheval.metrics.utility.metric_kolmogorov_smirnov import KolmogorovSmirnovTest

from midst_toolkit.evaluation.metrics_base import SynthEvalQualityMetric


class KolmogorovSmirnovAndTotalVariation(SynthEvalQualityMetric):
    def __init__(
        self,
        categorical_columns: list[str],
        numerical_columns: list[str],
        do_preprocess: bool = False,
        significance_level: float = 0.05,
        permutations: int = 1000,
    ):
        """
        This class performs a univariate comparison of corresponding columns in provided ``real_data`` and
        ``synthetic_data`` dataframes. The distribution of numerical columns is compared using a Kolmogorov-Smirnov
        (KS) test and categorical columns are compared with a Total Variation Distance (TVD) with significance
        established using a permutation test. Both a performed as two-sided hypothesis tests to determine whether it
        is likely that the distribution of a given column is the same between the two dataframes (null).

        The main score is the average test statistic across all evaluated columns. Smaller is better. Other scores
        returned include:

        - Average statistic and standard error thereof for numerical columns.
        - Average statistic and standard error thereof for categorical columns.
        - Average p-values for the statistics of all columns.
        - The number and percentage of columns that have statistically significant differences.

        Args:
            categorical_columns: Column names corresponding to the categorical variables of any provided dataframe. If
                no columns are provided, the associated stat values will be NaN
            numerical_columns: Column names corresponding to the numerical variables of any provided dataframe. If no
                columns are provided, the associated stat values will be NaN
            do_preprocess: Whether or not to preprocess the dataframes with the default pipeline used by SynthEval.
                Defaults to False.
            significance_level: Level of significance for the KS/TVD test statistics for a column of real vs. synthetic
                data to be considered significantly different. Lower implies a higher significance requirement.
            permutations: The number of permutations to run through to establish the TVD test statistic.
        """
        super().__init__(categorical_columns, numerical_columns, do_preprocess)
        self.significance_level = significance_level
        self.permutations = permutations
        self.all_columns = categorical_columns + numerical_columns

    def compute(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict[str, float]:
        """
        Compares the columns of ``real_data`` with those of ``synthetic_data`` pairwise with statistical tests. For
        numerical columns, this uses the Kolmogorov-Smirnov (KS) test and categorical columns are compared with a
        Total Variation Distance (TVD) with significance established using a permutation test.

        Args:
            real_data: Real data to which the synthetic data may be compared. In many cases this will be data used
                to TRAIN the model that generated the synthetic data, but not always.
            synthetic_data: Synthetically generated data whose quality is to be assessed.

        Returns:
            The results of both tests are combined into a single score and reported separately. These are keyed as
            follows:

            - 'avg stat', 'stat err': Average of all statistics (KS and TVD) and the standard error of the stats.
            - 'avg ks', 'ks err' : Average statistic and standard error thereof for numerical columns.
            - 'avg tvd', 'tvd err': Average statistic and standard error thereof for categorical columns.
            - 'avg pval', 'pval err': Average p-values for the statistics of all columns.
            - 'num sigs', 'frac sigs': The number and percentage of columns that have significance differences.
        """
        if self.do_preprocess:
            real_data, synthetic_data = self.preprocess(real_data, synthetic_data)

        # NOTE: The SynthEval KolmogorovSmirnovTest class ignores column specifications by default. However, for
        # other classes (correlation_matrix_difference for example), specifying less than all of the columns restricts
        # the score computation to just those columns. To make this consistent we do that here, before passing to the
        # SynthEval class.
        filtered_real_data = real_data[self.all_columns]
        filtered_synthetic_data = synthetic_data[self.all_columns]

        self.syntheval_metric = KolmogorovSmirnovTest(
            real_data=filtered_real_data,
            synt_data=filtered_synthetic_data,
            hout_data=None,
            cat_cols=self.categorical_columns,
            num_cols=self.numerical_columns,
            do_preprocessing=False,
            verbose=False,
        )

        return self.syntheval_metric.evaluate(sig_lvl=self.significance_level, n_perms=self.permutations)
