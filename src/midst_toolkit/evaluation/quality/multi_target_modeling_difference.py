import json
import multiprocessing as mp
import random
from collections import defaultdict
from logging import WARNING
from pathlib import Path
from statistics import mean
from typing import Any, Literal

import pandas as pd

from midst_toolkit.common.enumerations import ColumnType
from midst_toolkit.common.logger import log
from midst_toolkit.common.random import set_all_random_seeds
from midst_toolkit.evaluation.metrics_base import SynthEvalMetric
from midst_toolkit.evaluation.quality import MeanF1ScoreDifference, MeanRegressionDifference


ModelBasedMetric = MeanRegressionDifference | MeanF1ScoreDifference

METRIC_FILTER = (
    "avg_r2_difference",
    "avg_explained_variance_difference",
    "avg_mean_squared_error_difference",
    "avg_mean_absolute_error_difference",
    "avg_f1_difference",
)


def compute_for_single_label(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    holdout_data: pd.DataFrame,
    label_column_type: ColumnType,
    metric: MeanRegressionDifference | MeanF1ScoreDifference,
    random_seed: int | None = None,
) -> dict[str, float]:
    """
    This function is meant to facilitate evaluating on a single target column using a pre-constructed metric as part
    of a parallel set of processes in a multiprocessing pool.

    Args:
        real_data: Real data to which the synthetic data may be compared. In many cases this will be data used
            to TRAIN the model that generated the synthetic data, but not always.
        synthetic_data: Synthetically generated data whose quality is to be assessed.
        holdout_data: A real data with labels on which to measure the performance of the trained regression models
            performance. The holdout dataset should be preprocessed in the SAME WAY as the real and synthetic
            datasets. This must be provided for this metric. Defaults to None.
        label_column_type: The kind of target column we're modeling
        metric: The metric to be measured.
        random_seed: The random seed to use. If None provided then seeds will not be set in the processes. Defaults to
            None. NOTE: Seeds and randomness in multiprocessing is very annoying. This seed is a way for us to get
            consistent measurements when we fix a seed in the main code.

    Raises:
        ValueError: Will throw if the column type is not either numerical or categorical.

    Returns:
        The set of computed regression or classification metrics (depending on the column type) that were computed.
    """
    if random_seed is not None:
        set_all_random_seeds(random_seed)
    computed_metrics = metric.compute(real_data.copy(), synthetic_data.copy(), holdout_data.copy())
    if label_column_type == ColumnType.CATEGORICAL:
        # Categorical keys should include mean_f1_difference_holdout
        f1_difference = computed_metrics["mean_f1_difference_holdout"]
        return {"f1_difference": f1_difference}
    if label_column_type == ColumnType.NUMERICAL:
        return computed_metrics
    raise ValueError(f"Column type must be either NUMERICAL or CATEGORICAL. Received {label_column_type.value}")


class MultiTargetModelingDifference(SynthEvalMetric):
    def __init__(
        self,
        categorical_columns: list[str],
        numerical_columns: list[str],
        label_columns_and_type: dict[str, ColumnType],
        do_preprocess: bool = False,
        preprocess_labels: bool = False,
        folds: int = 5,
        f1_type: Literal["micro", "macro", "samples", "weighted", "binary"] = "micro",
        regressors_config_path: Path = Path("src/midst_toolkit/evaluation/quality/assets/regression_config.json"),
        measure_metrics_in_original_label_space: bool = False,
        include_regressor_specific_averages: bool = False,
        n_jobs: int = 1,
    ):
        """
        This class computes the difference in metrics for regression or classification models trained on real and
        synthetic data, depending on target column type. This is done over multiple target columns specified along
        with their type (categorical = classification, numerical = regression).

        Ideally, the synthetic data would be as effective at training a model to predict the target columns value as
        the real data. Note that this requires there to be a label column present for both the real and synthetic
        datasets.

        For regression, this class will train a set of models determined by the JSON file in the
        ``regressors_config_path``. This class leverages the functionality of the ``MeanRegressionDifference`` metric.
        Several metrics are reported to assess the quality of the trained models.

        For classification, this class will train several prediction models and compute F1 scores to assess
        performance. The differences in scores for real vs. synthetic data training are averaged across several
        models to produce the final difference value.

        The final scores include:

        - The average F1 score difference across all categorical targets.
        - The average regression score difference across all numerical targets for each kind of regression score.
        - The average F1 score and regression score difference across all target columns for each kind of regression
          score computed.

        For more details as to how classification and regression scores are computed see the
        ``MeanRegressionDifference`` and ``MeanF1ScoreDifference`` class documentation

        NOTE: A holdout set is REQUIRED for this metric. Preprocessing of the data is also important in getting the
        best assessment of the regressor or classification performance. This can be accomplished manually by
        preprocessing data before calling ``compute`` or using the default pipeline by setting ``do_preprocess`` to
        True. Note that if ``do_preprocess`` is True, the default pipelines for the ``MeanRegressionDifference`` and
        ``MeanF1ScoreDifference`` classes will be performed. For regression, no transformations to the label column
        are performed by default. In addition, preprocessing is performed independently on the real and synthetic
        data before fitting.

        Args:
            categorical_columns: Column names corresponding to the categorical variables of any provided dataframe.
            numerical_columns: Column names corresponding to the numerical variables of any provided dataframe.
            label_columns_and_type: A dictionary with column name keys and ColumnType values. The column names
                correspond to the targets for either regression or classification models to predict from the other
                columns in the dataset. If the ColumnType is NUMERICAL, regression models are trained. If it is
                CATEGORICAL, classification models are applied.
            do_preprocess: Whether or not to preprocess the dataframes with the default pipeline used by SynthEval.
                Defaults to False.
            preprocess_labels: Whether or not to preprocess the label column with a MinMaxScaler. This is only
                relevant for regression type tasks. Defaults to False.
            folds: Number of cross-validation folds for training/evaluating the set of classifiers used to
                establish a stable estimate of the classification difference. Only used for classification tasks.
                Defaults to 5.
            f1_type: The type of F1-score to be reported as the metric for classification tasks. The admissible
                values correspond to those of the sklearn implementation of ``f1_score``. Defaults to 'micro'.
            regressors_config_path: Path to the configuration file for the regressors to be applied in the evaluation.
                The default configuration (and a good example) are housed in the default path of this class.
                Defaults to Path("src/midst_toolkit/evaluation/quality/assets/regression_config.json").
            measure_metrics_in_original_label_space: Whether to transform labels into their original space prior to
                measuring metrics for regression tasks only. This only affects the metric measurements if
                ``preprocess_labels`` is set to True. Defaults to False.
            include_regressor_specific_averages: Whether to include the stats broken out by specific regressor models
                or only report the average regression metric across included regressors. Defaults to False.
            n_jobs: If greater than 1, this will attempt to perform the various regression or classification modeling
                tasks in parallel to speed up computation. This should specify the number of cpus available to
                perform computations. Defaults to 1.
        """
        super().__init__(categorical_columns, numerical_columns, do_preprocess)

        available_cores = mp.cpu_count()
        if n_jobs > available_cores:
            log(WARNING, f"Cores requested ({n_jobs}) exceeds cores available ({available_cores})")
        self.n_jobs = n_jobs

        assert len(label_columns_and_type) > 0, "No target columns supplied. The label_columns_and_type is empty."

        self.regressors_config_path = regressors_config_path
        self.include_regressor_specific_averages = include_regressor_specific_averages
        regressor_configs = self._get_regressors_specifications()
        self.label_columns_and_type = label_columns_and_type

        self.metrics: dict[str, ModelBasedMetric] = {}

        for label_column, column_type in self.label_columns_and_type.items():
            filtered_numerical_columns, filtered_categorical_columns = self.validate_label_column_and_filter(
                label_column, column_type
            )
            metric: ModelBasedMetric
            # If it's a numerical column, we perform regression
            if column_type == ColumnType.NUMERICAL:
                # If there is a special config for this column, we use it. Otherwise use the default.
                regressor_config = (
                    regressor_configs[label_column]
                    if label_column in regressor_configs
                    else regressor_configs["regressors"]
                )
                metric = MeanRegressionDifference(
                    categorical_columns=filtered_categorical_columns,
                    numerical_columns=filtered_numerical_columns,
                    label_column=label_column,
                    do_preprocess=do_preprocess,
                    preprocess_labels=preprocess_labels,
                    regressors_config=regressor_config,
                    include_additional_metrics=False,
                    measure_metrics_in_original_label_space=measure_metrics_in_original_label_space,
                )
            elif column_type == ColumnType.CATEGORICAL:
                metric = MeanF1ScoreDifference(
                    categorical_columns=filtered_categorical_columns,
                    numerical_columns=filtered_numerical_columns,
                    label_column=label_column,
                    do_preprocess=do_preprocess,
                    folds=folds,
                    f1_type=f1_type,
                )
            else:
                raise ValueError(f"Column type must be either NUMERICAL or CATEGORICAL. Received {column_type.value}")

            self.metrics[label_column] = metric

        self.measure_metrics_in_original_label_space = measure_metrics_in_original_label_space

    def _get_regressors_specifications(self) -> dict[str, list[dict[str, Any]]]:
        """
        Load the configurations file into a JSON structure. This can take two forms. The first is a set of regressors
        that will be applied for every classification task. These are specified at the top level of the config under
        the key "regressors." However, if a special set of regressors is desired for a particular column, the
        configuration can also include a key matching the target column with the same structure to include special
        settings for that specific column.

        NOTE: The configuration must always include a default set of configurations under the "regressors" key

        Returns:
            A dictionary with each entry being a list containing individual regression model configurations, including
            their sets of hyper-parameters to explore. The default set of regressors is under the "regressors" key. If
            any custom regressors were specified for individual columns, these are keyed by the column name to which
            they are to be applied.
        """
        with open(self.regressors_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        assert "regressors" in config, (
            f"Configuration file {self.regressors_config_path} must contain a 'regressors' key"
        )

        return config

    def validate_label_column_and_filter(
        self, label_column: str, column_type: ColumnType
    ) -> tuple[list[str], list[str]]:
        """
        Ensures that the label column is present in either the numerical or categorical columns provided. It will
        then remove the column from that list to be passed to either the regression or classification metric class.

        Args:
            label_column: Label column name for the regression or classification task.
            column_type: Type of the label column.

        Raises:
            ValueError: If we don't find the column in one of the two categories of columns.

        Returns:
            Filtered numerical and categorical columns with the label column removed.
        """
        if label_column in self.numerical_columns:
            assert column_type == ColumnType.NUMERICAL, "Label column is numerical but column_type is Categorical"
            return [column for column in self.numerical_columns if column != label_column], self.categorical_columns
        if label_column in self.categorical_columns:
            assert column_type == ColumnType.CATEGORICAL, "Label column is categorical but column_type is Numerical"
            return self.numerical_columns, [column for column in self.categorical_columns if column != label_column]
        raise ValueError(f"Label column: {label_column} is not present in designated columns")

    def compute(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, holdout_data: pd.DataFrame | None = None
    ) -> dict[str, float]:
        """
        This function computes the difference in metrics for regression or classification models trained on real and
        synthetic data, depending on target column type. This is done over multiple target columns specified along
        with their type (categorical = classification, numerical = regression).

        For regression, this class will train a set of models determined by the JSON file in the
        ``regressors_config_path``. This class leverages the functionality of the ``MeanRegressionDifference`` metric.
        Several metrics are reported to assess the quality of the trained models.

        For classification, this class will train several prediction models and compute F1 scores to assess
        performance. The differences in scores for real vs. synthetic data training are averaged across several
        models to produce the final difference value.

        The final scores include:

        - The average F1 score difference across all categorical targets.
        - The average regression score difference across all numerical targets for each kind of regression score.
        - The average F1 score and regression score difference across all target columns for each kind of regression
          score computed.

        For more details as to how classification and regression scores are computed see the
        ``MeanRegressionDifference`` and ``MeanF1ScoreDifference`` class documentation

        NOTE: A holdout set is REQUIRED for this metric. Preprocessing of the data is also important in getting the
        best assessment of the regressor or classification performance. This can be accomplished manually by
        preprocessing data before calling this function or using the default pipeline by setting ``do_preprocess`` to
        True in this class. Note that if ``do_preprocess`` is True, the default pipelines for the
        ``MeanRegressionDifference`` and ``MeanF1ScoreDifference`` classes will be performed. In addition,
        preprocessing is performed independently on the real and synthetic data before fitting.

        Args:
            real_data: Real data to which the synthetic data may be compared. In many cases this will be data used
                to TRAIN the model that generated the synthetic data, but not always.
            synthetic_data: Synthetically generated data whose quality is to be assessed.
            holdout_data: A real data with labels on which to measure the performance of the trained regression models
                performance. The holdout dataset should be preprocessed in the SAME WAY as the real and synthetic
                datasets. This must be provided for this metric. Defaults to None.

        Returns:
            - The average F1 score difference across all categorical targets.
            - The average regression score difference across all numerical targets for each kind of regression score.
            - The average F1 score and regression score difference across all target columns for each kind of
              regression score computed.
        """
        assert holdout_data is not None, "Multi-target analysis must have a holdout dataset"

        gathered_regression_differences: dict[str, list[float]] = defaultdict(list)
        gathered_f1_differences = []

        # Turn dictionary into a list of tuples for multiprocessing
        parameters_list = [
            (
                real_data,
                synthetic_data,
                holdout_data,
                self.label_columns_and_type[label_column],
                metric,
                int.from_bytes(random.randbytes(4)) if self.n_jobs > 1 else None,
            )
            for label_column, metric in self.metrics.items()
        ]

        if self.n_jobs == 1:
            # Using a pool is slightly slower if we don't want to parallelize. So we skip it.
            metrics_per_label = [compute_for_single_label(*parameters) for parameters in parameters_list]
        else:
            # This is required to address a hanging issue on linux machines. This forces MP to use spawning instead of
            # forking for all OSs. This is to avoid known hanging issues with MP.
            # See: https://britishgeologicalsurvey.github.io/science/python-forking-vs-spawn/
            multiprocessing_context = mp.get_context("spawn")
            with multiprocessing_context.Pool(self.n_jobs) as pool:
                metrics_per_label = pool.starmap(compute_for_single_label, parameters_list)

        # Post-process the metrics computed in parallel process
        for computed_metrics in metrics_per_label:
            for metric_name, metric_value in computed_metrics.items():
                if metric_name == "f1_difference":
                    gathered_f1_differences.append(metric_value)
                else:
                    gathered_regression_differences[metric_name].append(metric_value)

        # mean regression difference (per metric) across numerical target columns
        results = {
            metric_name: mean(metric_value) for metric_name, metric_value in gathered_regression_differences.items()
        }
        # mean f1 score difference across categorical target columns
        if len(gathered_f1_differences) > 0:
            results["avg_f1_difference"] = mean(gathered_f1_differences)
        # mean difference across all columns (broken our by regression type)
        results.update(
            {
                f"{metric_name}_and_f1_difference": mean(metric_value + gathered_f1_differences)
                for metric_name, metric_value in gathered_regression_differences.items()
            }
        )

        if not self.include_regressor_specific_averages:
            return {name: value for name, value in results.items() if name.startswith(METRIC_FILTER)}

        return results
