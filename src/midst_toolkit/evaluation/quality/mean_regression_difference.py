import json
from collections import defaultdict
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any, overload

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor  # noqa: F401
from sklearn.linear_model import LinearRegression  # noqa: F401
from sklearn.metrics import (
    explained_variance_score as compute_explained_variance,
)
from sklearn.metrics import (
    mean_absolute_error as compute_mean_absolute_error,
)
from sklearn.metrics import (
    mean_squared_error as compute_mean_squared_error,
)
from sklearn.metrics import (
    r2_score as compute_r2_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.neural_network import MLPRegressor  # noqa: F401
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tqdm import tqdm
from xgboost import XGBRegressor  # noqa: F401

from midst_toolkit.evaluation.metrics_base import SynthEvalMetric


class RegressorScores(Enum):
    R2 = "r2"
    EXPLAINED_VARIANCE = "explained_variance"
    MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
    MEAN_SQUARED_ERROR = "mean_squared_error"


class MeanRegressionDifference(SynthEvalMetric):
    def __init__(
        self,
        categorical_columns: list[str],
        numerical_columns: list[str],
        label_column: str,
        do_preprocess: bool = False,
        preprocess_labels: bool = False,
        regressors_config_path: Path = Path("src/midst_toolkit/evaluation/quality/assets/regression_config.json"),
        verbose: bool = True,
    ):
        """
        This class computes the difference in metrics for regression models trained on real and synthetic data.
        Ideally, the synthetic data would be as effective at training a regression model as the real data. Note that
        this requires there to be a regression label column present for both datasets. This class will train a set of
        regression models determined by the JSON file in the ``regressors_config_path``.

        The default configuration trains four sklearn models: ``LinearRegression``, ``MLPRegressor``,
        ``XGBRegressor``, and ``RandomForestRegressor``.

        The models are trained on provided real and synthetic data separately and these models are evaluated on a
        provided holdout set to establish regression performance metrics of r^2, explained variance, mean absolute
        error (MAE), and mean squared error (MSE). The configuration file can provide a range of hyper-parameters for
        these regressors. If so, a grid-search determines the most effective set of parameters for each of the
        metric-model pairs based on a validation set (split from the real or synthetic training data, respectively).

        This class reports:

        - Scores for each model-metric pair (for both real and synthetic training data)
        - Average scores across models for each metric (for both real and synthetic training data)
        - Average of all metric values across all models (for both real and synthetic training data)
        - Difference between scores for each model-metric pair (synthetic score - real score)
        - Difference between average scores across models for each metric (synthetic avg score - real avg score)
        - Difference in average of all metric value across all models (synthetic avg score - real avg score)

        For the differences and average differences, score interpretation varies depending on metric. For R^2 and
        explained variance, larger values (non-negative) are better. For MAE and MSE, smaller values are better
        (non-positive).

        NOTE: A holdout set is REQUIRED for this metric. Preprocessing of the data is also important in getting the
        best assessment of the regressor performance. This can be accomplished manually by preprocessing data before
        calling ``compute`` or using the default pipeline by setting ``do_preprocess`` to True. Note that if
        ``do_preprocess`` is True, the default pipeline performs MinMax scaling on the specified numerical columns
        and OneHotEncoding on the categorical columns. No transformations to the label column are performed by
        default. In addition, preprocessing is performed independently on the real and synthetic data before fitting.

        Args:
            categorical_columns: Column names corresponding to the categorical variables of any provided dataframe.
            numerical_columns: Column names corresponding to the numerical variables of any provided dataframe.
            label_column: Name of the column is the provided datasets that corresponds to the classification label to
                test dataset utility. This column MUST be present in both the real and synthetic data provided.
            do_preprocess: Whether or not to preprocess the dataframes with the default pipeline used by SynthEval.
                Defaults to False.
            preprocess_labels: Whether or not to preprocess the label column with a MinMaxScaler. Defaults to False.
            regressors_config_path: Path to the configuration file for the regressors to be applied in the evaluation.
                The default configuration (and a good example) are housed in the default path of this class.
                Defaults to Path("src/midst_toolkit/evaluation/quality/assets/regression_config.json").
            verbose: Whether or not to include the individual regressor performances in the metrics dictionary.
                If false, only the differences in performance will be returned. Defaults to True.
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
        self.regressors_config_path = regressors_config_path
        self.verbose = verbose
        self.preprocess_labels = preprocess_labels

    def get_regressors_specifications(self) -> list[dict[str, Any]]:
        """
        Load the configurations file into a JSON structure and extract the list of regressor specifications.

        Returns:
            A list containing individual regression model configurations, including their sets of hyper-parameters
            to explore.
        """
        with open(self.regressors_config_path, "r") as f:
            return json.load(f)["regressors"]

    def apply_transformations(
        self, dataset: pd.DataFrame, one_hot_encoder: OneHotEncoder | None, min_max_scaler: MinMaxScaler | None
    ) -> pd.DataFrame:
        """
        Given a dataset, this function applies a trained ``OneHotEncoder`` and/or ``MinMaxScaler`` to the categorical
        and numerical columns of the provided dataset, respectively. If one of the transformations is not provided,
        we assume that only columns processed with the provided transformation are meant to be returned. If neither
        are defined, we throw an error as it means no processing is done, which is likely not what the user intended.

        Args:
            dataset: Dataset to be transformed
            one_hot_encoder: Trained ``OneHotEncoder`` to transform the categorical columns of the dataset
            min_max_scaler: Trained ``MinMaxScaler`` to transform the numerical columns of the dataset

        Raises:
            ValueError: If both are ``one_hot_encoder`` and ``min_max_scaler`` are None, we throw an error as it means
            no processing is done, which is likely not what the user intended.

        Returns:
            A processed dataframe resulting from applying the provided transformations to the dataset.
        """
        # Assume that if a trained transformation is provided, at least one column is meant to be transformed.
        scaled_data = min_max_scaler.transform(dataset[self.numerical_columns]) if min_max_scaler is not None else None
        one_hot_data = (
            one_hot_encoder.transform(dataset[self.categorical_columns]) if one_hot_encoder is not None else None
        )

        if scaled_data is None and one_hot_data is None:
            raise ValueError("No columns to be encoded")
        if scaled_data is not None and one_hot_data is not None:
            preprocessed_data = pd.DataFrame(np.concatenate((scaled_data, one_hot_data), axis=1)).astype(float)
        elif one_hot_data is None:
            preprocessed_data = pd.DataFrame(scaled_data).astype(float)
        else:
            preprocessed_data = pd.DataFrame(one_hot_data).astype(float)

        # Adding back the target columns
        preprocessed_data[self.label_column] = dataset[self.label_column]
        return preprocessed_data

    def regression_preprocess(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply a default preprocessing pipeline fitted on the combined dataframes of ``train_data`` and ``test_data``.
        The default pipeline applies a MinMaxScaler to the numerical columns and a ``OneHotEncoder`` to the
        categorical columns.

        Args:
            train_data: Dataframe representing training data for a regression model.
            test_data: Dataframe representing testing data for evaluating a regression model.

        Returns:
            Train and test dataframes that have been preprocessed according to the default pipeline.
        """
        combined_data = pd.concat((train_data, test_data))

        if len(self.numerical_columns) > 0:
            combined_numerical_data = combined_data[self.numerical_columns]
            scaler = MinMaxScaler().fit(combined_numerical_data)
        else:
            scaler = None

        if len(self.categorical_columns) > 0:
            combined_categorical_data = combined_data[self.categorical_columns]
            one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
                combined_categorical_data
            )
        else:
            one_hot_encoder = None

        preprocessed_train_data = self.apply_transformations(train_data, one_hot_encoder, scaler)
        preprocessed_test_data = self.apply_transformations(test_data, one_hot_encoder, scaler)

        if self.preprocess_labels:
            label_scalar = MinMaxScaler().fit(combined_data[[self.label_column]])
            preprocessed_train_data[[self.label_column]] = label_scalar.transform(train_data[[self.label_column]])
            preprocessed_test_data[[self.label_column]] = label_scalar.transform(test_data[[self.label_column]])

        return preprocessed_train_data, preprocessed_test_data

    def prepare_training_data(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame, train_fraction: float = 0.9
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the provided training data into train and validation dataframes based on the fraction provided. This
        method will also preprocess the train and test data together if ``do_preprocess`` is True.

        Args:
            train_data: Training data to be split into train and test dataframes. Will be preprocessed prior to
                splitting if ``do_preprocess`` is True
            test_data: Test data will be preprocessed with the training data if ``do_preprocess`` is True. Otherwise
                it is unmodified.
            train_fraction: What percentage of the original training data ends up in the training data split. The
                remaining (``1-train_fraction``) ends up in the validation split  Defaults to 0.9.

        Returns:
            Training, validation, and test dataframes, potentially also preprocessed together.
        """
        assert 0 <= train_fraction <= 1, f"Train fraction is not in [0, 1]: {train_fraction}"

        if self.do_preprocess:
            processed_train_data, processed_test_data = self.regression_preprocess(train_data, test_data)
        else:
            processed_train_data = train_data.copy()
            processed_test_data = test_data.copy()

        split_train_data, split_validation_data = train_test_split(processed_train_data, train_size=train_fraction)
        return split_train_data, split_validation_data, processed_test_data

    @overload
    def train_and_evaluate_model(
        self,
        train_data_features: pd.DataFrame,
        train_data_labels: pd.DataFrame,
        test_data_features: pd.DataFrame,
        test_data_labels: pd.DataFrame,
        regressor: BaseEstimator,
    ) -> dict[str, float]: ...

    @overload
    def train_and_evaluate_model(
        self,
        train_data_features: pd.DataFrame,
        train_data_labels: pd.DataFrame,
        test_data_features: pd.DataFrame,
        test_data_labels: pd.DataFrame,
        regressor: BaseEstimator,
        compute_only: RegressorScores,
    ) -> float: ...

    def train_and_evaluate_model(
        self,
        train_data_features: pd.DataFrame,
        train_data_labels: pd.DataFrame,
        test_data_features: pd.DataFrame,
        test_data_labels: pd.DataFrame,
        regressor: BaseEstimator,
        compute_only: RegressorScores | None = None,
    ) -> float | dict[str, float]:
        """
        Provided a set of training features and labels and test data features and labels and a configured regression
        model, this function fits the provided model to the training data and then evaluates that model on the
        provided test data. By default, the model is evaluated on four metrics: R^2, explained variance, mean absolute
        error, and means squared error. If ``compute_only`` is provided, the dictionary of metrics will be limited
        to include only the metric specified by the argument.

        Args:
            train_data_features: Features on which to train the regressor model
            train_data_labels: Labels on which to train the regressor model
            test_data_features: Features on which to evaluate the trained regressor model
            test_data_labels: Labels on which to evaluate the trained regressor model
            regressor: A configured sklearn regression model to be fit on the training data and evaluated on the test
                data.
            compute_only: Determines whether to report only one metric from the set of four available. If None, all
                metrics will be computed. Defaults to None.

        Returns:
            A dictionary of all four metrics evaluated on the provided test data if ``compute_only`` is None. Otherwise
            a float representing only the metric specified by ``compute_only``.
        """
        regressor.fit(train_data_features, train_data_labels[self.label_column])
        pred = regressor.predict(test_data_features)

        if compute_only is None:
            r2 = compute_r2_score(test_data_labels, pred)
            explained_variance = compute_explained_variance(test_data_labels, pred)
            mean_squared_error = compute_mean_squared_error(test_data_labels, pred)
            mean_absolute_error = compute_mean_absolute_error(test_data_labels, pred)
            assert (
                isinstance(r2, float)
                and isinstance(explained_variance, float)
                and isinstance(mean_squared_error, float)
                and isinstance(mean_absolute_error, float)
            )
            return {
                "r2": r2,
                "explained_variance": explained_variance,
                "mean_squared_error": mean_squared_error,
                "mean_absolute_error": mean_absolute_error,
            }
        if compute_only == RegressorScores.R2:
            return compute_r2_score(test_data_labels, pred)
        if compute_only == RegressorScores.EXPLAINED_VARIANCE:
            return compute_explained_variance(test_data_labels, pred)
        if compute_only == RegressorScores.MEAN_ABSOLUTE_ERROR:
            return compute_mean_absolute_error(test_data_labels, pred)
        if compute_only == RegressorScores.MEAN_SQUARED_ERROR:
            return compute_mean_squared_error(test_data_labels, pred)
        raise ValueError(f"Unrecognized score option: {compute_only.value}")

    def tune_and_evaluate_regressor(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame, regressor_specifications: dict[str, Any]
    ) -> tuple[float, float, float, float]:
        """
        Provided training data, testing data, and specifications for a regression model, including hyper-parameters
        over which to tune the model, this function will find the best set of parameters, based on a validation set
        cut from ``train_data`` for each metric (R^2, explained variance, mean absolute error, and means squared
        error). Subsequently, this set of optimal parameters will be used to train a final model, separately for each
        metric, to be evaluated on the provided ``test_data``.

        Args:
            train_data: Training data for the regression model.
            test_data: Test data for evaluating the regression model.
            regressor_specifications: Specifications containing all information needed to configure the regression
                model, including sets of hyper-parameter to tune.

        Returns:
            The R^2, explained variance, mean absolute error, and means squared error scores for the regression model
            trained (and hyper-parameter tuned) on ``train_data`` when evaluated on `test_data`
        """
        # Create the regressor from the specifications
        model_class_str = regressor_specifications["class"]
        assert model_class_str in {"LinearRegression", "MLPRegressor", "XGBRegressor", "RandomForestRegressor"}
        model_class = eval(model_class_str)
        model_kwargs = regressor_specifications.get("kwargs", {})

        # Create a set of parameters to search
        parameter_set = list(ParameterGrid(model_kwargs))

        # Split the data and possibly preprocess
        train_data, validation_data, test_data = self.prepare_training_data(train_data, test_data)
        train_data_features = train_data.drop(self.label_column, axis=1, inplace=False)
        train_data_labels = train_data[[self.label_column]]
        validation_data_features = validation_data.drop(self.label_column, axis=1, inplace=False)
        validation_data_labels = validation_data[[self.label_column]]
        test_data_features = test_data.drop(self.label_column, axis=1, inplace=False)
        test_data_labels = test_data[[self.label_column]]

        # Run through all parameter combinations
        results = pd.DataFrame([])
        for parameters in tqdm(parameter_set):
            model = model_class(**parameters)

            metrics: dict[str, Any] = self.train_and_evaluate_model(
                train_data_features, train_data_labels, validation_data_features, validation_data_labels, model
            )

            metrics["parameters"] = parameters

            results = pd.concat((results, pd.DataFrame([metrics])))

        results.reset_index(inplace=True)
        best_r2_parameters = results.parameters[results.r2.idxmax()]
        best_explained_variance_parameters = results.parameters[results.explained_variance.idxmax()]
        best_mean_absolute_error_parameters = results.parameters[results.mean_absolute_error.idxmin()]
        best_mean_squared_error_parameters = results.parameters[results.mean_squared_error.idxmin()]

        # With each of the best parameter combinations by metric, we train a model with those parameters to be
        # evaluated on the test set.
        best_r2_score = self.train_and_evaluate_model(
            train_data_features,
            train_data_labels,
            test_data_features,
            test_data_labels,
            model_class(**best_r2_parameters),
            compute_only=RegressorScores.R2,
        )
        best_explained_variance_score = self.train_and_evaluate_model(
            train_data_features,
            train_data_labels,
            test_data_features,
            test_data_labels,
            model_class(**best_explained_variance_parameters),
            compute_only=RegressorScores.EXPLAINED_VARIANCE,
        )
        best_mean_absolute_error_score = self.train_and_evaluate_model(
            train_data_features,
            train_data_labels,
            test_data_features,
            test_data_labels,
            model_class(**best_mean_absolute_error_parameters),
            compute_only=RegressorScores.MEAN_ABSOLUTE_ERROR,
        )
        best_mean_squared_error_score = self.train_and_evaluate_model(
            train_data_features,
            train_data_labels,
            test_data_features,
            test_data_labels,
            model_class(**best_mean_squared_error_parameters),
            compute_only=RegressorScores.MEAN_SQUARED_ERROR,
        )
        return (
            best_r2_score,
            best_explained_variance_score,
            best_mean_absolute_error_score,
            best_mean_squared_error_score,
        )

    def process_results(self, results: dict[str, dict[str, float]]) -> dict[str, float]:
        """
        A post-processing function to flatten the provided results dictionary and compute other statistics, such
        as average metric performance across regression models and average performance across all models and metrics
        combined.

        Args:
            results: The results to be postprocess, they are first keyed by regression model name, then by the various
                computed metrics "r2", "explained_variance", "mean_squared_error", and "mean_absolute_error".

        Returns:
            A flattened dictionary of results where metric model pairs are keyed by "{model_name}_{metric_name}"
            metric averages across models are keyed by "avg_{metric_name}" and average across all models and metrics
            as "avg_all_scores".
        """
        # package real data results
        processed_results = {}
        all_scores = []
        gathered_metrics = defaultdict(list)
        for model_name, model_results in results.items():
            for metric_name, score in model_results.items():
                processed_results[f"{model_name}_{metric_name}"] = score
                gathered_metrics[metric_name].append(score)
                all_scores.append(score)
            for metric_name, scores in gathered_metrics.items():
                processed_results[f"avg_{metric_name}"] = mean(scores)

        processed_results["avg_all_scores"] = mean(all_scores)
        return processed_results

    def package_all_results(
        self, real_data_results: dict[str, dict[str, float]], synthetic_data_results: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """
        A post-processing function to properly structure the results of training regression models and real and
        synthetic data and evaluating the resulting models on a holdout set. The results dictionaries are first
        flattened and additional statistics are computed according to `process_results`. Then, the differences
        in the various metrics when training on synthetic vs. real data are computed and stored as well.

        If ``self.verbose`` is ``True``, then performance of the individual regression models when trained on real or
        synthetic data are also reported. If it is false, only the difference between a metrics value for synthetic
        vs. real training data is returned.

        Args:
            real_data_results: Metrics associated with training regression models on real data.
            synthetic_data_results: Metrics associated with training regression models on real data.

        Returns:
            A set of results for the difference between each metric when regression models are trained on synthetic vs.
            real data. If ``self.verbose`` is ``True``, this also includes individual metrics for real and synthetic
            training data, respectively.
        """
        processed_real_results = self.process_results(real_data_results)
        processed_synthetic_results = self.process_results(synthetic_data_results)
        merged_scores = {}

        assert processed_real_results.keys() == processed_synthetic_results.keys(), (
            "Metrics keys for real and synthetic data should be equal"
        )

        # Assumption is that the metric keys are completely shared.
        for metric_name, real_score in processed_real_results.items():
            synthetic_score = processed_synthetic_results[metric_name]
            merged_scores[f"{metric_name}_difference"] = synthetic_score - real_score

        # Need to prefix the shared metric keys so they do not overwrite each other
        if self.verbose:
            merged_scores.update({f"real_{key}": value for key, value in processed_real_results.items()})
            merged_scores.update({f"synthetic_{key}": value for key, value in processed_synthetic_results.items()})

        return merged_scores

    def compute(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, holdout_data: pd.DataFrame | None = None
    ) -> dict[str, float]:
        """
        This function computes the difference in metrics for regression models trained on real and synthetic data.
        Ideally, the synthetic data would be as effective at training a regression model as the real data. Note that
        this requires there to be a regression label column present for both datasets. The regression models to be
        trained are determined by the JSON file in the ``regressors_config_path`` of the class.

        The default configuration trains four sklearn models: ``LinearRegression``, ``MLPRegressor``,
        ``XGBRegressor``, and ``RandomForestRegressor``.

        The models are trained on provided real and synthetic data separately and these models are evaluated on a
        provided holdout set to establish regression performance metrics of r^2, explained variance, mean absolute
        error (MAE), and mean squared error (MSE). The configuration file can provide a range of hyper-parameters for
        these regressors. If so, a grid-search determines the most effective set of parameters for each of the
        metric-model pairs based on a validation set (split from the real or synthetic training data, respectively).

        This function reports:

        - Scores for each model-metric pair (for both real and synthetic training data)
        - Average scores across models for each metric (for both real and synthetic training data)
        - Average of all metric values across all models (for both real and synthetic training data)
        - Difference between scores for each model-metric pair (synthetic score - real score)
        - Difference between average scores across models for each metric (synthetic avg score - real avg score)
        - Difference in average of all metric value across all models (synthetic avg score - real avg score)

        For the differences and average differences, score interpretation varies depending on metric. For R^2 and
        explained variance, larger values (non-negative) are better. For MAE and MSE, smaller values are better
        (non-positive).

        NOTE: Preprocessing of the data is also important in getting the best assessment of the regressor performance.
        This can be accomplished manually by preprocessing data before calling this function or using the default
        pipeline by setting ``do_preprocess`` to True. Note that if ``do_preprocess`` is True, the default pipeline
        performs MinMax scaling on the specified numerical columns and OneHotEncoding on the categorical columns. No
        transformations to the label column are performed by default. In addition, preprocessing is performed
        independently on the real and synthetic data before fitting.

        Args:
            real_data: Real data to which the synthetic data may be compared. In many cases this will be data used
                to TRAIN the model that generated the synthetic data, but not always.
            synthetic_data: Synthetically generated data whose quality is to be assessed.
            holdout_data: A real data with labels on which to measure the performance of the trained regression models
                performance. The holdout dataset should be preprocessed in the SAME WAY as the real and synthetic
                datasets. This must be provided for this metric. Defaults to None.

        Returns:
            Metrics associated with the difference in regression performance when training on real vs. synthetic data.
        """
        assert holdout_data is not None, "Regression analysis must have a holdout dataset"
        assert is_float_dtype(real_data[self.label_column]), "Label column must have a float type for regression"
        assert is_float_dtype(synthetic_data[self.label_column]), "Label column must have a float type for regression"
        assert is_float_dtype(holdout_data[self.label_column]), "Label column must have a float type for regression"

        filtered_real_data = real_data[self.all_columns]
        filtered_synthetic_data = synthetic_data[self.all_columns]
        filtered_holdout_data = holdout_data[self.all_columns]

        assert self.label_column in filtered_real_data.columns, (
            f"Label column: {self.label_column} must be in real_data"
        )
        assert self.label_column in filtered_synthetic_data.columns, (
            f"Label column: {self.label_column} must be in synthetic_data"
        )

        assert self.label_column in filtered_holdout_data.columns, (
            f"Label column: {self.label_column} must be in holdout_data"
        )

        all_regressor_specifications = self.get_regressors_specifications()

        # First we train using the real data and test on the holdout set for all regressors
        real_data_scores: dict[str, dict[str, float]] = {}
        for regressor_specifications in all_regressor_specifications:
            model_name = regressor_specifications["class"]
            r2_score, explained_variance, mean_absolute_error, mean_squared_error = self.tune_and_evaluate_regressor(
                filtered_real_data, filtered_holdout_data, regressor_specifications
            )
            real_data_scores[model_name] = {
                "r2": r2_score,
                "explained_variance": explained_variance,
                "mean_squared_error": mean_squared_error,
                "mean_absolute_error": mean_absolute_error,
            }

        # Next we train using the synthetic data and test on the holdout set
        synthetic_data_scores: dict[str, dict[str, float]] = {}
        for regressor_specifications in all_regressor_specifications:
            model_name = regressor_specifications["class"]
            r2_score, explained_variance, mean_absolute_error, mean_squared_error = self.tune_and_evaluate_regressor(
                filtered_synthetic_data, filtered_holdout_data, regressor_specifications
            )
            synthetic_data_scores[model_name] = {
                "r2": r2_score,
                "explained_variance": explained_variance,
                "mean_squared_error": mean_squared_error,
                "mean_absolute_error": mean_absolute_error,
            }

        return self.package_all_results(real_data_scores, synthetic_data_scores)
