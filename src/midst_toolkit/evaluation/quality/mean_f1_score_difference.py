import pandas as pd
from syntheval.metrics.utility.metric_accuracy_difference import ClassificationAccuracy as SynthEvalF1ScoreDifference

from midst_toolkit.evaluation.metrics_base import SynthEvalMetric


def extract_from_dictionary(
    result: dict[str, float | dict[str, float]],
    upper_key: str,
    lower_key: str | None = None,
) -> float:
    """
    This function is mostly to appease mypy. This extracts values from the result dictionary in a strongly typed way.
    That is, this handles the ambiguity associated with the mix of nested and not nested dictionaries in result. There
    are two settings.

    - An upper key along is provided. If so, we assume the target value is NOT nested and enforce that the value
      extracted at the first level is, indeed, a float.
    - Both an upper and lower key are provided. We assume that we are reaching into a nested dictionary and extract
      a float value accordingly.

    Args:
        result: Set of single- and two-level dictionaries we're grabbing a result from.
        upper_key: First level key.
        lower_key: Second level key if the value we're reaching for is part of a nested dictionary. Defaults to None.

    Returns:
        Float value associated with the upper (and possible lower keys) key.
    """
    value = result[upper_key]
    if lower_key is not None:
        assert isinstance(value, dict)
        return value[lower_key]
    assert isinstance(value, float)
    return value


def post_process_results(
    result: dict[str, float | dict[str, float]], process_holdout: bool = False
) -> dict[str, float]:
    """
    This function is meant to flatten the results dictionaries returned by SynthEval. SynthEval returns nested
    dictionaries to summarize the performance of the individual models. This is useful to know for someone evaluated
    the synthetic data, but we flatten the dictionary to make it simpler.

    Args:
        result: The nested dictionaries result returned by SynthEval.
        process_holdout: Whether or not a set of results for a holdout dataset is to be processed. If a holdout
            dataset was not evaluated but this is set to True, an error will throw. Defaults to False.

    Returns:
        Flattened version of the nested dictionary of results with a few items dropped for clarity, like standard error
        of individual classifier performances.
    """
    flat_result = {
        "random_forest_real_train_f1": extract_from_dictionary(result, "rf", "rr_val_acc"),
        "random_forest_synthetic_train_f1": extract_from_dictionary(result, "rf", "fr_val_acc"),
        "adaboost_real_train_f1": extract_from_dictionary(result, "adaboost", "rr_val_acc"),
        "adaboost_synthetic_train_f1": extract_from_dictionary(result, "adaboost", "fr_val_acc"),
        "svm_real_train_f1": extract_from_dictionary(result, "svm", "rr_val_acc"),
        "svm_synthetic_train_f1": extract_from_dictionary(result, "svm", "fr_val_acc"),
        "logreg_real_train_f1": extract_from_dictionary(result, "logreg", "rr_val_acc"),
        "logreg_synthetic_train_f1": extract_from_dictionary(result, "logreg", "fr_val_acc"),
        "mean_f1_difference": extract_from_dictionary(result, "avg diff"),
        "f1_difference_standard_error": extract_from_dictionary(result, "avg diff err"),
    }
    if process_holdout:
        flat_result["random_forest_real_train_f1_holdout"] = extract_from_dictionary(result, "rf", "rr_test_acc")
        flat_result["random_forest_synthetic_train_f1_holdout"] = extract_from_dictionary(result, "rf", "fr_test_acc")
        flat_result["adaboost_real_train_f1_holdout"] = extract_from_dictionary(result, "adaboost", "rr_test_acc")
        flat_result["adaboost_synthetic_train_f1_holdout"] = extract_from_dictionary(result, "adaboost", "fr_test_acc")
        flat_result["svm_real_train_f1_holdout"] = extract_from_dictionary(result, "svm", "rr_test_acc")
        flat_result["svm_synthetic_train_f1_holdout"] = extract_from_dictionary(result, "svm", "fr_test_acc")
        flat_result["logreg_real_train_f1_holdout"] = extract_from_dictionary(result, "logreg", "rr_test_acc")
        flat_result["logreg_synthetic_train_f1_holdout"] = extract_from_dictionary(result, "logreg", "fr_test_acc")
        flat_result["mean_f1_difference_holdout"] = extract_from_dictionary(result, "avg diff hout")
        flat_result["f1_difference_standard_error_holdout"] = extract_from_dictionary(result, "avg diff err hout")
    return flat_result


class MeanF1ScoreDifference(SynthEvalMetric):
    def __init__(
        self,
        categorical_columns: list[str],
        numerical_columns: list[str],
        label_column: str,
        do_preprocess: bool = False,
        folds: int = 5,
        f1_type: str = "micro",
    ):
        """
        This class computes the difference in F1 score for classifiers trained on real and synthetic data. Ideally,
        the synthetic data would be as effective at training a classifier as the real data. Note that this requires
        there to be a classification label column present for both datasets. This class will train four classifiers
        (scikit-learn's decision tree, AdaBoost, random forest, and logistic regression) on the provided real and
        synthetic data separately via cross-validation folds and test the resulting classifiers performance on the
        validation set of real data. The average difference in performance across the folds is measured. This
        average difference is then averaged to get the final score.

        If a holdout set is provided, the same process is repeated but the whole real and test datasets are used
        to train models (i.e. no cross-validation for this measurement) and their performance measured on the holdout
        set. The differences in performance are then averaged.

        A value close to zero for this metric is better. A negative value implies that they synthetic data is worse
        on average at training the classifiers. A positive value (while likely rare) means it is better.

        NOTE: Categorical variables need to be encoded before training the classifier. This can be accomplished by
        preprocessing before calling ``compute`` or by setting ``do_preprocess`` to True. Note that if
        ``do_preprocess`` is True, the default Syntheval pipeline is used, which does NOT one-hot encode the
        categoricals.

        NOTE: Despite the naming convention of the SynthEval metrics class, the metric is not accuracy but rather F1
        score (of some kind).

        Args:
            categorical_columns: Column names corresponding to the categorical variables of any provided dataframe.
            numerical_columns: Column names corresponding to the numerical variables of any provided dataframe.
            label_column: Name of the column is the provided datasets that corresponds to the classification label to
                test dataset utility. This column MUST be present in both the real and synthetic data provided.
            do_preprocess: Whether or not to preprocess the dataframes with the default pipeline used by SynthEval.
                Defaults to False.
            folds: Number of cross-validation folds for training/evaluating the set of classifiers used to
                establish a stable estimate of the classification difference. Defaults to 5.
            f1_type: The type of F1-score to be reported as the metric. The admissible values correspond to those of
                the sklearn implementation of ``f1_score``. Defaults to 'micro'.
        """
        super().__init__(categorical_columns, numerical_columns, do_preprocess)
        assert label_column not in numerical_columns, (
            "Label column should not be included in the set of numerical columns provided"
        )
        assert label_column not in categorical_columns, (
            "Label column should not be included in the set of numerical columns provided"
        )
        self.label_column = label_column
        self.all_columns = categorical_columns + numerical_columns + [label_column]
        self.folds = folds
        self.f1_type = f1_type

    def compute(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, holdout_data: pd.DataFrame | None = None
    ) -> dict[str, float]:
        """
        This function computes the difference in F1 score for classifiers trained on real and synthetic data. Ideally,
        the synthetic data would be as effective at training a classifier as the real data. Note that this requires
        there to be a classification label column present for both datasets. This class will train four classifiers
        (scikit-learn's decision tree, AdaBoost, random forest, and logistic regression) on the provided real and
        synthetic data separately via cross-validation folds and test the resulting classifiers performance on the
        validation set of real data. The average difference in performance across the folds is measured. This
        average difference is then averaged to get the final score.

        If a holdout set is provided, the same process is repeated but the whole real and test datasets are used
        to train models (i.e. no cross-validation for this measurement) and their performance measured on the holdout
        set. The differences in performance are then averaged.

        NOTE: Categorical variables need to be encoded before training the classifier. This can be accomplished by
        preprocessing before calling ``compute`` or by setting ``do_preprocess`` to True. Note that if
        ``do_preprocess`` is True, the default Syntheval pipeline is used, which does NOT one-hot encode the
        categoricals.

        NOTE: Despite the naming convention of the SynthEval metrics class, the metric is not accuracy but rather F1
        score (of some kind).

        Args:
            real_data: Real data to which the synthetic data may be compared. In many cases this will be data used
                to TRAIN the model that generated the synthetic data, but not always.
            synthetic_data: Synthetically generated data whose quality is to be assessed.
            holdout_data: Defaults to None.

        Returns:
            A dictionary with the performance results for the various trained classifiers and the mean f1 difference
            score. The most important score is keyed under 'mean_f1_difference', which is the difference in the
            classifier performance when trained on real and synthetic data applied to classifier other real data.
            For more details on the contents on the dictionary see the ``post_process_results`` function.
        """
        if self.do_preprocess:
            if holdout_data is None:
                real_data, synthetic_data = self.preprocess(real_data, synthetic_data)
            else:
                real_data, synthetic_data, holdout_data = self.preprocess(real_data, synthetic_data, holdout_data)

        # NOTE: The SynthEval MutualInformation class ignores column specifications by default. However, for
        # other classes (correlation_matrix_difference for example), specifying less than all of the columns restricts
        # the score computation to just those columns. To make this consistent we do that here, before passing to the
        # SynthEval class.
        filtered_real_data = real_data[self.all_columns]
        filtered_synthetic_data = synthetic_data[self.all_columns]
        filtered_holdout_data = holdout_data[self.all_columns] if holdout_data is not None else None

        assert self.label_column in filtered_real_data.columns, (
            f"Label column: {self.label_column} must be in real_data"
        )
        assert self.label_column in filtered_synthetic_data.columns, (
            f"Label column: {self.label_column} must be in synthetic_data"
        )

        self.syntheval_metric = SynthEvalF1ScoreDifference(
            real_data=filtered_real_data,
            synt_data=filtered_synthetic_data,
            hout_data=filtered_holdout_data,
            # SynthEval wants cat_cols to have the analysis target (label) included so we jam it in.
            cat_cols=self.categorical_columns + [self.label_column],
            num_cols=self.numerical_columns,
            do_preprocessing=False,
            verbose=False,
            analysis_target=self.label_column,
        )
        result = self.syntheval_metric.evaluate(F1_type=self.f1_type, k_folds=self.folds, full_output=False)
        return post_process_results(result, process_holdout=(holdout_data is not None))
