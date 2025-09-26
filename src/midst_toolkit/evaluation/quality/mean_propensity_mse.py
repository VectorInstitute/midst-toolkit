import pandas as pd
from syntheval.metrics.utility.metric_propensity_mse import PropensityMeanSquaredError

from midst_toolkit.evaluation.metrics_base import SynthEvalQualityMetric


class MeanPropensityMeanSquaredError(SynthEvalQualityMetric):
    def __init__(
        self,
        categorical_columns: list[str],
        numerical_columns: list[str],
        do_preprocess: bool = False,
        folds: int = 5,
        max_iterations: int = 100,
        solver: str = "liblinear",
    ):
        """
        This class measures how well a ``LogisticRegression`` model from sklearn (as implemented in SynthEval) can
        distinguish between real and synthetic data. The classification model is trained on a subset of the two data
        sources and then applied to a validation split of the mixed data, created through cross-validation folds. The
        average pMSE for synthetic vs. real predictions and macro F1 scores across the folds are reported along with
        the standard error of these mean values.

        Computation of pMSE is based on the formula in:

        Woo, M., Reiter, J.P., Oganian, A., Karr, A.F.: Global measures of data utility for microdata masked for
        disclosure limitation. J. Priv. Confidentiality 1(1) (2009) https://doi.org/10.29012/jpc.v1i1.568

        NOTE: Categorical variables need to be encoded before training the classifier. This can be accomplished by
        preprocessing before calling ``compute`` or by setting ``do_preprocess`` to True. Note that if
        ``do_preprocess`` is True, the default Syntheval pipeline is used, which does NOT one-hot encode the
        categoricals.

        - A smaller pMSE is better. In cases where the two datasets are balanced in size, 0.25 is worst case.
        - Higher Macro F1 is better.

        Args:
            categorical_columns: Column names corresponding to the categorical variables of any provided dataframe.
            numerical_columns: Column names corresponding to the numerical variables of any provided dataframe.
            do_preprocess: Whether or not to preprocess the dataframes with the default pipeline used by SynthEval.
                Defaults to False.
            folds: Number of cross-validation folds for training/evaluating the LogisticRegression classifier used to
                establish a stable estimate of the pMSE. Defaults to 5.
            max_iterations: Maximum number of iterations for the regression fitting. Defaults to 100.
            solver: Kind of solver used to fit the ``LogisticRegression`` model. Options coincide with those of the
                sklearn ``LogisticRegression`` implementation. Defaults to 'liblinear'.
        """
        super().__init__(categorical_columns, numerical_columns, do_preprocess)
        self.all_columns = categorical_columns + numerical_columns
        self.folds = folds
        self.max_iterations = max_iterations
        self.solver = solver

    def compute(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict[str, float]:
        """
        Computes how well a LogisticRegression model from sklearn (as implemented in SynthEval) can distinguish between
        real and synthetic data. The classification model is trained on a subset of the two data sources and then
        applied to a validation split of the mixed data, created through cross-fold validation on the combination of
        the two datasets. The average pMSE of the 0 = synthetic, 1 = real predictions and macro F1 scores across the
        folds are reported along with the standard error of these mean values.

        NOTE: Categorical variables need to be encoded before training the classifier. This can be accomplished by
        preprocessing before calling ``compute`` or by setting ``do_preprocess`` to True. Note that if
        ``do_preprocess`` is True, the default Syntheval pipeline is used, which does NOT one-hot encode the
        categoricals.

        Args:
            real_data: Real data to which the synthetic data may be compared. In many cases this will be data used
                to TRAIN the model that generated the synthetic data, but not always.
            synthetic_data: Synthetically generated data whose quality is to be assessed.

        Returns:
            The mean pMSE and macro F1 scores for a LogisticRegression model. These values are keyed by 'avg_pmse' and
            'avg_macro_f1_score' respectively. The standard errors associated with these mean values are reported under
            the keys 'pmse_standard_error' and 'macro_f1_standard_error' as well.
        """
        if self.do_preprocess:
            real_data, synthetic_data = self.preprocess(real_data, synthetic_data)

        # NOTE: The SynthEval MutualInformation class ignores column specifications by default. However, for
        # other classes (correlation_matrix_difference for example), specifying less than all of the columns restricts
        # the score computation to just those columns. To make this consistent we do that here, before passing to the
        # SynthEval class.
        filtered_real_data = real_data[self.all_columns]
        filtered_synthetic_data = synthetic_data[self.all_columns]

        # Syntheval also ASSUMES you don't have a column in both provided dataframes called 'real' because it will
        # attach another column with the same name, so we throw an error here if the column already exists.
        assert "real" not in filtered_real_data.columns, "A column called 'real' already exists in the dataframe."
        assert "real" not in filtered_synthetic_data.columns, "A column called 'real' already exists in the dataframe."

        self.syntheval_metric = PropensityMeanSquaredError(
            real_data=filtered_real_data,
            synt_data=filtered_synthetic_data,
            hout_data=None,
            cat_cols=self.categorical_columns,
            num_cols=self.numerical_columns,
            do_preprocessing=False,
            verbose=False,
        )
        result = self.syntheval_metric.evaluate(self.folds, self.max_iterations, self.solver)
        result["avg_pmse"] = result.pop("avg pMSE")
        result["pmse_standard_error"] = result.pop("pMSE err")
        result["avg_macro_f1_score"] = result.pop("avg acc")
        result["macro_f1_standard_error"] = result.pop("acc err")
        return result
