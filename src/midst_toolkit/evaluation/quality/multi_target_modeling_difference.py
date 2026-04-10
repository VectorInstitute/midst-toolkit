import json
import multiprocessing as mp
import random
from collections import defaultdict
from logging import WARNING, INFO
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
    # MODIFICATION: Add individual score filters
    "avg_real_r2",
    "avg_synthetic_r2",
    "avg_real_f1",
    "avg_synthetic_f1",
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
        
        # MODIFICATION: Extract individual F1 scores
        result = {"f1_difference": f1_difference}
        
        if "real_f1_holdout" in computed_metrics:
            result["f1_real"] = computed_metrics["real_f1_holdout"]
        if "synthetic_f1_holdout" in computed_metrics:
            result["f1_synthetic"] = computed_metrics["synthetic_f1_holdout"]
            
        log(INFO, f"F1 Metrics computed: {list(computed_metrics.keys())}")
        
        return result
    
    if label_column_type == ColumnType.NUMERICAL:
        # MODIFICATION: The regression metrics already include individual scores when
        # include_additional_metrics=True, so we just pass them through
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
        MODIFIED VERSION: Configured to use ONLY Random Forest for both regression and classification.
        
        This class computes the difference in metrics for regression or classification models trained on real and
        synthetic data, depending on target column type. This is done over multiple target columns specified along
        with their type (categorical = classification, numerical = regression).

        NOTE: To use only Random Forest:
        1. For regression: Pass a custom regressors_config_path pointing to regression_config_RF_ONLY.json
        2. For classification: This version automatically uses only Random Forest

        Args:
            categorical_columns: Column names corresponding to the categorical variables of any provided dataframe.
            numerical_columns: Column names corresponding to the numerical variables of any provided dataframe.
            label_columns_and_type: A dictionary with column name keys and ColumnType values.
            do_preprocess: Whether or not to preprocess the dataframes. Defaults to False.
            preprocess_labels: Whether or not to preprocess the label column. Defaults to False.
            folds: Number of cross-validation folds. Defaults to 5.
            f1_type: The type of F1-score to be reported. Defaults to 'micro'.
            regressors_config_path: Path to the configuration file for regressors. 
                USE regression_config_RF_ONLY.json for only Random Forest.
            measure_metrics_in_original_label_space: Whether to transform labels. Defaults to False.
            include_regressor_specific_averages: Whether to include individual regressor scores. Defaults to False.
            n_jobs: Number of parallel workers. Defaults to 1.
        """
        super().__init__(categorical_columns, numerical_columns, do_preprocess)
        self.label_columns_and_type = label_columns_and_type
        self.preprocess_labels = preprocess_labels
        self.folds = folds
        self.f1_type = f1_type
        self.measure_metrics_in_original_label_space = measure_metrics_in_original_label_space
        self.regressors_config_path = regressors_config_path
        self.include_regressor_specific_averages = include_regressor_specific_averages
        self.n_jobs = n_jobs

        # Create all the metrics for each label with appropriate column type
        self.metrics: dict[str, MeanRegressionDifference | MeanF1ScoreDifference] = {}
        for label_column, column_type in label_columns_and_type.items():
            filtered_numerical_columns, filtered_categorical_columns = self.validate_label_column_and_filter(
                label_column, column_type
            )
            if column_type == ColumnType.CATEGORICAL:
                # MODIFICATION: Use the RF-only version of MeanF1ScoreDifference
                self.metrics[label_column] = MeanF1ScoreDifference(
                    filtered_categorical_columns,
                    filtered_numerical_columns,
                    label_column,
                    do_preprocess=do_preprocess,
                    folds=folds,
                    f1_type=f1_type,
                )
            elif column_type == ColumnType.NUMERICAL:
                # MODIFICATION: Set include_additional_metrics=True to get individual scores
                # Use regressors_config_path that points to RF-only config
                self.metrics[label_column] = MeanRegressionDifference(
                    filtered_categorical_columns,
                    filtered_numerical_columns,
                    label_column,
                    do_preprocess=do_preprocess,
                    preprocess_labels=preprocess_labels,
                    regressors_config=regressors_config_path,
                    include_additional_metrics=True,  # MODIFICATION: Get individual scores
                    measure_metrics_in_original_label_space=measure_metrics_in_original_label_space,
                )
            else:
                raise ValueError(
                    f"Column type must be numerical or categorical. Received: {column_type.value} for column: "
                    f"{label_column}"
                )

    def get_regressors_specifications(self) -> list[dict[str, Any]]:
        """
        Load the regressor specifications from the JSON configuration file.

        Returns:
            A dictionary of regressors specifications.
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
        Ensures that the label column is present in either the numerical or categorical columns provided.

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
        Compute metrics using ONLY Random Forest for both regression and classification.

        Args:
            real_data: Real training data.
            synthetic_data: Synthetic data to be evaluated.
            holdout_data: Holdout test data (required).

        Returns:
            Dictionary containing difference metrics and individual scores.
        """
        assert holdout_data is not None, "Multi-target analysis must have a holdout dataset"

        gathered_regression_differences: dict[str, list[float]] = defaultdict(list)
        gathered_f1_differences = []
        
        # MODIFICATION: Add storage for individual scores
        gathered_real_r2_scores = []
        gathered_synthetic_r2_scores = []
        gathered_real_f1_scores = []
        gathered_synthetic_f1_scores = []

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
            metrics_per_label = [compute_for_single_label(*parameters) for parameters in parameters_list]
        else:
            multiprocessing_context = mp.get_context("spawn")
            with multiprocessing_context.Pool(self.n_jobs) as pool:
                metrics_per_label = pool.starmap(compute_for_single_label, parameters_list)

        # Post-process the metrics computed in parallel process
        for computed_metrics in metrics_per_label:
            for metric_name, metric_value in computed_metrics.items():
                if metric_name == "f1_difference":
                    gathered_f1_differences.append(metric_value)
                # MODIFICATION: Capture individual F1 scores
                elif metric_name == "f1_real":
                    gathered_real_f1_scores.append(metric_value)
                elif metric_name == "f1_synthetic":
                    gathered_synthetic_f1_scores.append(metric_value)
                # MODIFICATION: Capture individual R2 scores
                elif metric_name == "real_avg_r2":
                    gathered_real_r2_scores.append(metric_value)
                elif metric_name == "synthetic_avg_r2":
                    gathered_synthetic_r2_scores.append(metric_value)
                else:
                    gathered_regression_differences[metric_name].append(metric_value)

        # mean regression difference (per metric) across numerical target columns
        results = {
            metric_name: mean(metric_value) for metric_name, metric_value in gathered_regression_differences.items()
        }
        
        # MODIFICATION: Add individual R2 scores to results
        if len(gathered_real_r2_scores) > 0:
            results["avg_real_r2"] = mean(gathered_real_r2_scores)
        if len(gathered_synthetic_r2_scores) > 0:
            results["avg_synthetic_r2"] = mean(gathered_synthetic_r2_scores)
        
        # mean f1 score difference across categorical target columns
        if len(gathered_f1_differences) > 0:
            results["avg_f1_difference"] = mean(gathered_f1_differences)
            
        # MODIFICATION: Add individual F1 scores to results
        if len(gathered_real_f1_scores) > 0:
            results["avg_real_f1"] = mean(gathered_real_f1_scores)
        if len(gathered_synthetic_f1_scores) > 0:
            results["avg_synthetic_f1"] = mean(gathered_synthetic_f1_scores)
            
        # mean difference across all columns
        results.update(
            {
                f"{metric_name}_and_f1_difference": mean(metric_value + gathered_f1_differences)
                for metric_name, metric_value in gathered_regression_differences.items()
            }
        )

        if not self.include_regressor_specific_averages:
            return {name: value for name, value in results.items() if name.startswith(METRIC_FILTER)}

        return results