"""Defines the dataset functions for the ClavaDDPM model."""

import hashlib
import json
import pickle
from collections import Counter
from copy import deepcopy
from dataclasses import astuple, dataclass, replace
from logging import INFO
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import pandas as pd
import torch
from category_encoders import LeaveOneOutEncoder
from scipy.special import expit, softmax
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileTransformer,
    StandardScaler,
)

from midst_toolkit.common.enumerations import DataSplit, PredictionType, TaskType
from midst_toolkit.common.logger import log
from midst_toolkit.models.clavaddpm.enumerations import (
    ArrayDict,
    CategoricalEncoding,
    CategoricalNaNPolicy,
    IsTargetCondioned,
    Normalization,
    NumericalNaNPolicy,
    TargetPolicy,
)


# Wildcard value to which all rare categorical variables are mapped
CAT_RARE_VALUE = "_rare_"
CAT_MISSING_VALUE = "_nan_"


@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Normalization | None = None
    numerical_nan_policy: NumericalNaNPolicy | None = None
    categorical_nan_policy: CategoricalNaNPolicy | None = None
    category_minimum_frequency: float | None = None
    categorical_encoding: CategoricalEncoding | None = CategoricalEncoding.ORDINAL
    target_policy: TargetPolicy | None = TargetPolicy.DEFAULT

    @classmethod
    def default(cls) -> Self:
        """Return the default transformations."""
        return cls(seed=0, normalization=Normalization.QUANTILE, target_policy=TargetPolicy.DEFAULT)


@dataclass(frozen=False)
class Dataset:
    x_num: ArrayDict | None
    x_cat: ArrayDict | None
    y: ArrayDict
    y_info: dict[str, Any]
    task_type: TaskType
    n_classes: int | None
    categorical_transform: OneHotEncoder | None = None
    numerical_transform: StandardScaler | None = None

    @classmethod
    def from_dir(cls, directory: Path) -> Self:
        """
        Load a dataset from a directory.

        Args:
            directory: The directory to load the dataset from. Can be a Path object or a path string.

        Returns:
            The loaded dataset.
        """
        if Path(directory / "info.json").exists():
            info = json.loads(Path(directory / "info.json").read_text())

        return cls(
            cls._load_datasets(directory, "x_num") if directory.joinpath("x_num_train.npy").exists() else None,
            cls._load_datasets(directory, "x_cat") if directory.joinpath("x_cat_train.npy").exists() else None,
            cls._load_datasets(directory, "y"),
            {},
            TaskType(info["task_type"]),
            info.get("n_classes"),
        )

    @classmethod
    def _load_datasets(cls, directory: Path, dataset_name: str) -> ArrayDict:
        """
        Load all the dataset splits from a directory.

        Will check which of the splits exist in the directory for the
        given dataset_name and load all of them.

        Args:
            directory: The directory to load the dataset from.
            dataset_name: The dataset_name to load.

        Returns:
            The loaded datasets with all the splits.
        """
        splits = [k.value for k in list(DataSplit) if directory.joinpath(f"y_{k.value}.npy").exists()]
        # TODO: figure out if there is a way of getting rid of the cast
        return {x: cast(np.ndarray, np.load(directory / f"{dataset_name}_{x}.npy", allow_pickle=True)) for x in splits}

    @property
    def is_binclass(self) -> bool:
        """
        Check if the dataset is a binary classification dataset.

        Returns:
            True if the dataset is a binary classification dataset, False otherwise.
        """
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        """
        Check if the dataset is a multiclass classification dataset.

        Returns:
            True if the dataset is a multiclass classification dataset, False otherwise.
        """
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        """
        Check if the dataset is a regression dataset.

        Returns:
            True if the dataset is a regression dataset, False otherwise.
        """
        return self.task_type == TaskType.REGRESSION

    @property
    def n_num_features(self) -> int:
        """
        Get the number of numerical features in the dataset.

        That number should be in the second dimension of the tensors of x_num.

        Returns:
            The number of numerical features in the dataset.
        """
        return 0 if self.x_num is None else self.x_num[DataSplit.TRAIN.value].shape[1]

    @property
    def n_cat_features(self) -> int:
        """
        Get the number of categorical features in the dataset.

        That number should be in the second dimension of the tensors of x_cat.

        Returns:
            The number of categorical features in the dataset.
        """
        return 0 if self.x_cat is None else self.x_cat[DataSplit.TRAIN.value].shape[1]

    @property
    def n_features(self) -> int:
        """
        Get the total number of features in the dataset.

        Returns:
            The total number of features in the dataset.
        """
        return self.n_num_features + self.n_cat_features

    def size(self, split: DataSplit | None) -> int:
        """
        Get the size of a dataset split. If no split is provided, the size of
        the entire dataset is returned.

        Args:
            split: The split of the dataset to get the size of.
                If None, the size of the entire dataset is returned.

        Returns:
            The size of the dataset.
        """
        return sum(map(len, self.y.values())) if split is None else len(self.y[split.value])

    @property
    def output_dimension(self) -> int:
        """
        Get the output dimension of the model.

        For self.task_type == TaskType.MULTICLASS, the output dimension is the number of classes.
        For self.task_type == TaskType.REGRESSION, the output dimension is 1.
        For self.task_type == TaskType.BINCLASS, the output dimension is also 1 because it is label encoded.

        Returns:
            The output dimension of the model.
        """
        if self.is_multiclass:
            assert self.n_classes is not None
            return self.n_classes
        return 1

    def get_category_sizes(self, split: DataSplit) -> list[int]:
        """
        Get the size of the categories in the specified split of the dataset.

        Args:
            split: The split of the dataset to get the size of the categories of.

        Returns:
            The size of the categories in the specified split of the dataset.
        """
        return [] if self.x_cat is None else get_category_sizes(self.x_cat[split.value])

    def calculate_metrics(
        self,
        predictions: dict[str, np.ndarray],
        prediction_type: PredictionType | None,
    ) -> dict[str, Any]:
        """
        Calculate the metrics of the predictions.

        Args:
            predictions: The predictions to calculate the metrics of.
            prediction_type: The type of the predictions.

        Returns:
            The metrics of the predictions.
        """
        metrics = {
            x: calculate_metrics(self.y[x], predictions[x], self.task_type, prediction_type, self.y_info)
            for x in predictions
        }
        if self.task_type == TaskType.REGRESSION:
            score_key = "rmse"
            score_sign = -1
        else:
            score_key = "accuracy"
            score_sign = 1
        for part_metrics in metrics.values():
            part_metrics["score"] = score_sign * part_metrics[score_key]
        return metrics


# TODO consider moving all the functions below into the Dataset class
def get_category_sizes(features: torch.Tensor | np.ndarray) -> list[int]:
    """
    Get the size of the categories in the data by counting the number of
    unique values in each column.

    Args:
        features: The data from which to extract category sizes.

    Returns:
        A list with the category sizes in the data.
    """
    x_t = features.T.cpu().tolist() if isinstance(features, torch.Tensor) else features.T.tolist()
    return [len(set(xt)) for xt in x_t]


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: TaskType,
    prediction_type: PredictionType | None,
    y_info: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate the metrics of the predictions.

    Usage: calculate_metrics(y_true, y_pred, TaskType.BINCLASS, PredictionType.LOGITS, {})

    Args:
        y_true: The true labels as a numpy array.
        y_pred: The predicted labels as a numpy array.
        task_type: The type of the task.
        prediction_type: The type of the predictions.
        y_info: A dictionary with metadata about the labels.

    Returns:
        The metrics of the predictions as a dictionary with the following keys:
            If the task type is TaskType.REGRESSION:
                {
                    "rmse": The root mean squared error.
                    "r2": The R^2 score.
                }

            If the task type is TaskType.MULTICLASS, it will have a key for each label
            with the following metrics (result of sklearn.metrics.classification_report):
                {
                    "label-1": {
                        "precision": The precision of the label.
                        "recall": The recall of the label.
                        "f1-score": The F1 score of the label.
                        "support": The number of occurrences of this label in y_true.
                    },
                    "label-2": {...}
                    ...
                }

            If the task type is TaskType.BINCLASS, it will have a key for each label
            with the following metrics ((result of sklearn.metrics.classification_report),
            and an additional ROC AUC metric:
                {
                    "label-1": {
                        "precision": The precision of the label.
                        "recall": The recall of the label.
                        "f1-score": The F1 score of the label.
                        "support": The number of occurrences of this label in y_true.
                    },
                    "label-2": {...}
                    ...
                    "roc_auc": The ROC AUC score.
                }
    """
    if task_type == TaskType.REGRESSION:
        assert prediction_type is None
        assert "std" in y_info
        rmse = calculate_rmse(y_true, y_pred, y_info["std"])
        r2 = r2_score(y_true, y_pred)
        result = {"rmse": rmse, "r2": r2}
    else:
        labels, probs = _get_predicted_labels_and_probs(y_pred, task_type, prediction_type)
        # TODO: figure out if there is a way of getting rid of the cast
        result = cast(dict[str, Any], classification_report(y_true, labels, output_dict=True))
        if task_type == TaskType.BINCLASS:
            result["roc_auc"] = roc_auc_score(y_true, probs)
    return result


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray, std: float | None) -> float:
    """
    Calculate the root mean squared error (RMSE) of the predictions.

    Args:
        y_true: The true labels as a numpy array.
        y_pred: The predicted labels as a numpy array.
        std: The standard deviation of the labels. If None, the RMSE is calculated
            without the standard deviation.

    Returns:
        The RMSE of the predictions.
    """
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    if std is not None:
        rmse *= std
    return rmse


def _get_predicted_labels_and_probs(
    y_pred: np.ndarray, task_type: TaskType, prediction_type: PredictionType | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Get the labels and probabilities from the predictions.
    If prediction_type is None, will return the predicted labels as is
    and the probabilities as None.

    Args:
        y_pred: The predicted labels as a numpy array.
        task_type: The type of the task. Can be TaskType.BINCLASS or TaskType.MULTICLASS.
            Other task types are not supported.
        prediction_type: The type of the predictions. If None, will return the predictions as labels
            and probabilities as None.

    Returns:
        A tuple with the labels and probabilities. The probabilities are None
            if the prediction_type is None.
    """
    assert task_type in (TaskType.BINCLASS, TaskType.MULTICLASS), f"Unsupported task type: {task_type.value}"

    if prediction_type is None:
        return y_pred, None

    if prediction_type == PredictionType.LOGITS:
        probs = expit(y_pred) if task_type == TaskType.BINCLASS else softmax(y_pred, axis=1)
    elif prediction_type == PredictionType.PROBS:
        probs = y_pred
    else:
        raise ValueError(f"Unsupported prediction_type: {prediction_type.value}")

    assert probs is not None
    labels = np.round(probs) if task_type == TaskType.BINCLASS else probs.argmax(axis=1)
    return labels.astype("int64"), probs


def make_dataset_from_df(
    data: pd.DataFrame,
    transformations: Transformations,
    is_target_conditioned: IsTargetCondioned,
    info: dict[str, Any],
    data_split_ratios: list[float] | None = None,
    noise_scale: float = 0,
    data_split_random_state: int = 42,
) -> tuple[Dataset, dict[int, LabelEncoder], list[str]]:
    """
    Generate a dataset from a pandas DataFrame.

    The order of the generated dataset: (y, x_num, x_cat).

    Note: For now, n_classes has to be set to 0. This is because our matrix is the concatenation
    of (x_num, x_cat). In this case, if we have is_y_cond == 'concat', we can guarantee that y
    is the first column of the matrix.
    However, if we have n_classes > 0, then y is not the first column of the matrix.

    Args:
        data: The pandas DataFrame to generate the dataset from.
        transformations: The transformations to apply to the dataset.
        is_target_conditioned: The condition on the y column.
            IsTargetCondioned.CONCAT: y is concatenated to X, the model learns a joint distribution of (y, X)
            IsTargetCondioned.EMBEDDING: y is not concatenated to X. During computations, y is embedded
                and added to the latent vector of X
            IsTargetCondioned.NONE: y column is completely ignored

            How does is_target_conditioned affect the generation of y?
            is_target_conditioned:
                IsTargetCondioned.CONCAT: the model synthesizes (y, X) directly, so y is just the first column
                IsTargetCondioned.EMBEDDING: y is first sampled using empirical distribution of y. The model only
                    synthesizes X. When returning the generated data, we return the generated X
                    and the sampled y. (y is sampled from empirical distribution, instead of being
                    generated by the model)
                    Note that in this way, y is still not independent of X, because the model has been
                    adding the embedding of y to the latent vector of X during computations.
                IsTargetCondioned.NONE:
                    y is synthesized using y's empirical distribution. X is generated by the model.
                    In this case, y is completely independent of X.

        info: A dictionary with metadata about the DataFrame.
        data_split_ratios: The ratios of the dataset to split into train, val, and test. The sum of
            the ratios must amount to 1 (with a tolerance of 0.01). Optional, default is [0.7, 0.2, 0.1].
        noise_scale: The scale of the noise to add to the categorical features. Optional, default is 0.
        data_split_random_state: The random state to use for the data split. Will be passed down to the
            train_test_split function from sklearn. Optional, default is 42.

    Returns:
        A tuple with the dataset, the label encoders, and the column names in the order they appear in the dataset.
    """
    if data_split_ratios is None:
        data_split_ratios = [0.7, 0.2, 0.1]

    assert len(data_split_ratios) == 3, "The ratios must be a list of 3 values (train, validation, test)."
    assert np.isclose(sum(data_split_ratios), 1, atol=0.01), (
        "The sum of the ratios must amount to 1 (with a tolerance of 0.01)."
    )

    train_val_data, test_data = train_test_split(
        data,
        test_size=data_split_ratios[2],
        random_state=data_split_random_state,
    )
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=data_split_ratios[1] / (data_split_ratios[0] + data_split_ratios[1]),
        random_state=data_split_random_state,
    )

    categorical_column_names, numerical_column_names = _get_categorical_and_numerical_column_names(
        info,
        is_target_conditioned,
    )

    if len(categorical_column_names) > 0:
        categorical_features = {
            DataSplit.TRAIN.value: train_data[categorical_column_names].to_numpy(dtype=np.str_),
            DataSplit.VALIDATION.value: val_data[categorical_column_names].to_numpy(dtype=np.str_),
            DataSplit.TEST.value: test_data[categorical_column_names].to_numpy(dtype=np.str_),
        }
    else:
        categorical_features = None

    if len(numerical_column_names) > 0:
        numerical_features = {
            DataSplit.TRAIN.value: train_data[numerical_column_names].values.astype(np.float32),
            DataSplit.VALIDATION.value: val_data[numerical_column_names].values.astype(np.float32),
            DataSplit.TEST.value: test_data[numerical_column_names].values.astype(np.float32),
        }
    else:
        numerical_features = None

    target = {
        DataSplit.TRAIN.value: train_data[info["y_col"]].values.astype(np.float32),
        DataSplit.VALIDATION.value: val_data[info["y_col"]].values.astype(np.float32),
        DataSplit.TEST.value: test_data[info["y_col"]].values.astype(np.float32),
    }

    # build the column_orders list
    # It's a string list with the names numerical columns followed by the names of
    # the categorical columns in order they appear in the dataset that will be returned
    index_to_column = list(data.columns)
    column_to_index = {col: i for i, col in enumerate(index_to_column)}
    categorical_column_orders = [column_to_index[col] for col in categorical_column_names]
    numerical_column_orders = [column_to_index[col] for col in numerical_column_names]

    column_orders_indices = numerical_column_orders + categorical_column_orders
    column_orders = [index_to_column[index] for index in column_orders_indices]

    # Encode the categorical features and merge them with the numerical features
    numerical_features, label_encoders = _encode_and_merge_features(
        categorical_features,
        numerical_features,
        noise_scale,
    )

    assert isinstance(info["n_classes"], int)

    dataset = Dataset(
        numerical_features,
        None,
        target,
        y_info={},
        task_type=TaskType(info["task_type"]),
        n_classes=info["n_classes"],
    )

    return transform_dataset(dataset, transformations, None), label_encoders, column_orders


def _get_categorical_and_numerical_column_names(
    info: dict[str, Any],
    is_target_conditioned: IsTargetCondioned,
) -> tuple[list[str], list[str]]:
    """
    Get the categorical and numerical column names from the info dictionary.

    Args:
        info: The info dictionary.
        is_target_conditioned: The condition on the y column.
    """
    numerical_columns: list[str] = []
    categorical_columns: list[str] = []

    if info["n_classes"] > 0:
        if info["cat_cols"] is not None:
            categorical_columns += info["cat_cols"]
        if is_target_conditioned == IsTargetCondioned.CONCAT:
            categorical_columns += [info["y_col"]]

        numerical_columns = info["num_cols"]

    else:
        if info["num_cols"] is not None:
            numerical_columns += info["num_cols"]
        if is_target_conditioned == IsTargetCondioned.CONCAT:
            numerical_columns += [info["y_col"]]

        categorical_columns = info["cat_cols"]

    return categorical_columns, numerical_columns


def _encode_and_merge_features(
    categorical_features: ArrayDict | None,
    numerical_features: ArrayDict | None,
    noise_scale: float,
) -> tuple[ArrayDict, dict[int, LabelEncoder]]:
    """
    Merge the categorical with the numerical features for train, validation, and test datasets.

    The categorical features are encoded and then merged with the numerical features. The
    label encoders used to do that are also returned.

    If ``noise_scale`` is greater than 0, noise from a normal distribution with a standard
    deviation of ``noise_scale`` is added to the categorical features.

    Args:
        categorical_features: A dictionary with the categorical features data for train,
            validation, and test datasets.
        numerical_features: A dictionary with the numerical features data for train,
            validation, and test datasets.
        noise_scale: The scale of the noise to add to the categorical features.

    Returns:
        The merged features for train, validation, and test datasets and the label encoders
        used to do so.
    """
    if categorical_features is None:
        # if no categorical features, just return the numerical features
        assert numerical_features is not None
        return numerical_features, {}

    # Otherwise, encode the categorical features
    all_categorical_data = np.vstack(
        (
            categorical_features[DataSplit.TRAIN.value],
            categorical_features[DataSplit.VALIDATION.value],
            categorical_features[DataSplit.TEST.value],
        )
    )

    categorical_data_converted = []
    label_encoders = {}
    for column in range(all_categorical_data.shape[1]):
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(all_categorical_data[:, column]).astype(float)
        categorical_data_converted.append(encoded_labels)
        if noise_scale > 0:
            # add noise
            categorical_data_converted[-1] += np.random.normal(0, noise_scale, categorical_data_converted[-1].shape)
        label_encoders[column] = label_encoder

    categorical_data_transposed = np.vstack(categorical_data_converted).T

    num_train_samples = categorical_features[DataSplit.TRAIN.value].shape[0]
    num_validation_samples = categorical_features[DataSplit.VALIDATION.value].shape[0]

    categorical_features[DataSplit.TRAIN.value] = categorical_data_transposed[:num_train_samples, :]
    categorical_features[DataSplit.VALIDATION.value] = categorical_data_transposed[
        num_train_samples : num_train_samples + num_validation_samples, :
    ]
    categorical_features[DataSplit.TEST.value] = categorical_data_transposed[
        num_train_samples + num_validation_samples :, :
    ]

    if numerical_features is None:
        # if no numerical features then no need to merge, just return the categorical features
        return categorical_features, label_encoders

    # Otherwise, merge the categorical and numerical features
    merged_features = {
        DataSplit.TRAIN.value: np.concatenate(
            (numerical_features[DataSplit.TRAIN.value], categorical_features[DataSplit.TRAIN.value]), axis=1
        ),
        DataSplit.VALIDATION.value: np.concatenate(
            (numerical_features[DataSplit.VALIDATION.value], categorical_features[DataSplit.VALIDATION.value]),
            axis=1,
        ),
        DataSplit.TEST.value: np.concatenate(
            (numerical_features[DataSplit.TEST.value], categorical_features[DataSplit.TEST.value]), axis=1
        ),
    }

    return merged_features, label_encoders


def transform_dataset(
    dataset: Dataset,
    transformations: Transformations,
    cache_dir: Path | None,
) -> Dataset:
    """
    Transform the dataset.

    Args:
        dataset: The dataset to transform.
        transformations: The transformations to apply to the dataset.
        cache_dir: The directory to cache the transformed dataset.
            Optional, default is None. If not None, will check if the transformations exist in the cache directory.
            If they do, will returned the cached transformed dataset. If not, will transform the dataset and cache it.

    Returns:
        The transformed dataset.
    """
    # WARNING: the order of transformations matters. Moreover, the current
    # implementation is not ideal in that sense.
    cache_path = None
    if cache_dir is not None:
        # if cache_dir is not None, will save the cache file path into the cache_path variable
        # so the transformations can be saved in the cache dir
        transformations_md5 = hashlib.md5(str(transformations).encode("utf-8")).hexdigest()
        transformations_str = "__".join(map(str, astuple(transformations)))
        cache_path = cache_dir / f"cache__{transformations_str}__{transformations_md5}.pickle"
        if cache_path.exists():
            cache_transformations, value = load_pickle(cache_path)
            if transformations == cache_transformations:
                print(f"Using cached features: {cache_dir.name + '/' + cache_path.name}")
                return value
            raise RuntimeError(f"Hash collision for {cache_path}")

    if dataset.x_num is not None:
        dataset = process_nans_in_numerical_features(dataset, transformations.numerical_nan_policy)

    numerical_transform = None
    categorical_transform = None
    numerical_features = dataset.x_num
    categorical_features = dataset.x_cat

    if numerical_features is not None and transformations.normalization is not None:
        numerical_features, numerical_transform = normalize(
            numerical_features,
            transformations.normalization,
            transformations.seed,
        )

    if categorical_features is None:
        assert transformations.categorical_nan_policy is None
        assert transformations.category_minimum_frequency is None
    else:
        categorical_features = process_nans_in_categorical_features(
            categorical_features,
            transformations.categorical_nan_policy,
        )
        if transformations.category_minimum_frequency is not None:
            categorical_features = collapse_rare_categories(
                categorical_features,
                transformations.category_minimum_frequency,
            )

        categorical_features, is_numerical, categorical_transform = encode_categorical_features(
            categorical_features,
            transformations.categorical_encoding,
            dataset.y[DataSplit.TRAIN.value],
            transformations.seed,
            return_encoder=True,
        )
        if is_numerical:
            if numerical_features is None:
                numerical_features = categorical_features
            else:
                numerical_features = {
                    x: np.hstack([numerical_features[x], categorical_features[x]]) for x in numerical_features
                }
            categorical_features = None

    target, target_info = build_target(dataset.y, transformations.target_policy, dataset.task_type)

    dataset = replace(dataset, x_num=numerical_features, x_cat=categorical_features, y=target, y_info=target_info)
    dataset.numerical_transform = numerical_transform
    dataset.categorical_transform = categorical_transform

    if cache_path is not None:
        dump_pickle((transformations, dataset), cache_path)

    return dataset


def load_pickle(path: Path | str, **kwargs: Any) -> Any:
    """
    Load a pickle file.

    Args:
        path: The path to the pickle file.
        **kwargs: Additional arguments to pass to the pickle.loads function.

    Returns:
        The loaded pickle file.
    """
    return pickle.loads(Path(path).read_bytes(), **kwargs)


def dump_pickle(x: Any, path: Path | str, **kwargs: Any) -> None:
    """
    Dump an object into a pickle file.

    Args:
        x: The object to dump.
        path: The path to the pickle file.
        **kwargs: Additional arguments to pass to the pickle.dumps function.
    """
    Path(path).write_bytes(pickle.dumps(x, **kwargs))


# Inspired by: https://github.com/yandex-research/rtdl/blob/a4c93a32b334ef55d2a0559a4407c8306ffeeaee/lib/data.py#L20
def normalize(
    x: ArrayDict,
    normalization: Normalization,
    seed: int | None,
) -> tuple[ArrayDict, StandardScaler | MinMaxScaler | QuantileTransformer]:
    """
    Normalize the input data.

    Args:
        x: The data to normalize.
        normalization: The normalization to use.
        seed: The seed to use for the random state. Optional, default is None.

    Returns:
        The normalized data and the normalizer.
    """
    x_train = x[DataSplit.TRAIN.value]
    if normalization == Normalization.STANDARD:
        normalizer = StandardScaler()
    elif normalization == Normalization.MINMAX:
        normalizer = MinMaxScaler()
    elif normalization == Normalization.QUANTILE:
        normalizer = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(x[DataSplit.TRAIN.value].shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported normalization: {normalization.value}")
    normalizer.fit(x_train)

    return {k: normalizer.transform(v) for k, v in x.items()}, normalizer


def process_nans_in_numerical_features(dataset: Dataset, policy: NumericalNaNPolicy | None) -> Dataset:
    """
    Process the NaN values in the numerical features of the dataset. Note that the signature here is different from
    that of ``process_nans_in_categorical_features``, because, if we are dropping rows with NaN values, we need to
    also remove the corresponding categorical rows from the dataset.

    Args:
        dataset: The dataset to process.
        policy: The policy to use to process the NaN values for the numerical features.

    Returns:
        The processed dataset.
    """
    if policy is None:
        log(INFO, "No NaN processing policy specified.")
        return dataset

    assert dataset.x_num is not None, "No numerical features are present to process."

    nan_masks = {k: np.isnan(v) for k, v in dataset.x_num.items()}
    nan_values_exist = any(mask.any() for mask in nan_masks.values())
    if not nan_values_exist:
        log(INFO, "No NaN values to be processed.")
        return dataset

    if policy == NumericalNaNPolicy.DROP_ROWS:
        # mapping from datasplit key to 1D boolean array with entries corresponding to rows of an array. An entry of
        # True indicates no columns of that row have a NaN entry. False implies at least 1 column entry does.
        valid_masks = {k: ~v.any(1) for k, v in nan_masks.items()}
        # Test set should not have NaNs
        assert valid_masks[DataSplit.TEST.value].all(), (
            "Cannot drop test rows, since this will affect the final metrics."
        )

        dataset.x_num = None if dataset.x_num is None else drop_rows_according_to_mask(dataset.x_num, valid_masks)
        dataset.x_cat = None if dataset.x_cat is None else drop_rows_according_to_mask(dataset.x_cat, valid_masks)
        dataset.y = drop_rows_according_to_mask(dataset.y, valid_masks)

    elif policy == NumericalNaNPolicy.MEAN:
        # Computes column means in the training dataset, ignoring NaN values.
        new_values = np.nanmean(dataset.x_num[DataSplit.TRAIN.value], axis=0)

        # If any training column is all-NaN, np.nanmean returns NaN
        bad_cols = np.isnan(new_values)
        if bad_cols.any():
            raise ValueError("At least one of the columns in the train split are all NaN")

        numerical_features_per_split = deepcopy(dataset.x_num)
        for data_split, numerical_features in numerical_features_per_split.items():
            nan_indices = np.where(nan_masks[data_split])
            numerical_features[nan_indices] = np.take(new_values, nan_indices[1])
        dataset.x_num = numerical_features_per_split
    else:
        raise ValueError(f"Unsupported policy: {policy.value}")

    return dataset


def drop_rows_according_to_mask(data_split: ArrayDict, valid_masks: dict[str, np.ndarray]) -> ArrayDict:
    """
    Provided a dictionary of keys to numpy arrays, this function drops rows in each numpy array in the dictionary
    according to the values in `valid_masks`. The keys of `valid_masks` must match the entries in data.

    Args:
        data_split: The data to apply the mask to.
        valid_masks: Mapping from datasplit key to 1D boolean array with entries corresponding to rows of an array.
            An entry of True indicates that the row should be kept. False implies it should be dropped.

    Returns:
        The data with the mask applied, dropping rows corresponding to False entries of the mask.
    """
    if set(data_split.keys()) != set(valid_masks.keys()):
        raise KeyError("Keys of data do not match the provided valid_masks")
    # Dropping rows in each array that have a False entry in valid_masks
    filtered_data_split: ArrayDict = {}
    for split_name, data in data_split.items():
        row_mask = valid_masks[split_name]
        if row_mask.ndim != 1 or row_mask.shape[0] != data.shape[0]:
            raise ValueError(f"Mask for split '{split_name}' has shape {row_mask.shape}; expected ({data.shape[0]},)")
        filtered_data_split[split_name] = data[row_mask]
    return filtered_data_split


def process_nans_in_categorical_features(data_splits: ArrayDict, policy: CategoricalNaNPolicy | None) -> ArrayDict:
    """
    Process the NaN values in the categorical features of the datasets provided. Supports only string or float arrays.

    Args:
        data_splits: A dictionary containing data to process, split into different partitions. One of which must
            be keys with DataSplit.TRAIN.value.
        policy: The policy to use to process the NaN values. If none, will no-op.

    Returns:
        The processed data.
    """
    if policy is None:
        log(INFO, "No NaN processing policy specified.")
        return data_splits

    assert len(data_splits) > 0, "data_splits is empty, processing will fail."

    # Determine whether the arrays are float or string typed. We assume all arrays in data_splits have the same type
    train_data_split = data_splits[DataSplit.TRAIN.value]
    is_float_array = np.issubdtype(train_data_split.dtype, np.floating)
    # Value that we're looking for to replace
    missing_values = float("nan") if is_float_array else CAT_MISSING_VALUE

    # If there are any NaN values, try to apply a the policy.
    nan_values = [
        np.isnan(data).any() if is_float_array else (data == CAT_MISSING_VALUE).any() for data in data_splits.values()
    ]
    if any(nan_values):
        if policy == CategoricalNaNPolicy.MOST_FREQUENT:
            imputer = SimpleImputer(missing_values=missing_values, strategy=policy.value)
            imputer.fit(data_splits[DataSplit.TRAIN.value])
            return {k: imputer.transform(v) for k, v in data_splits.items()}
        raise ValueError(f"Unsupported cat_nan_policy: {policy.value}")

    # If no nan values are present. We do nothing.
    return data_splits


def collapse_rare_categories(data_splits: ArrayDict, min_frequency: float) -> ArrayDict:
    """
    Collapses rare categories in each column of the datasets under ``data_splits`` into a single category encoded by
    the global variable CAT_RARE_VALUE. Categories considered rare are those not satisfying the ``min_frequency``
    threshold within the training split of ``data_splits``.

    NOTE: Arrays must be of type string

    Args:
        data_splits: A dictionary containing data to process, split into different partitions. One of which must
            be keys with DataSplit.TRAIN.value.
        min_frequency: The minimum frequency threshold of the categories to keep. Has to be between 0 and 1.

    Returns:
        The processed data.
    """
    assert 0.0 < min_frequency < 1.0, "min_frequency has to be between 0 and 1"

    training_data = data_splits[DataSplit.TRAIN.value]
    min_count = max(1, int(np.ceil(len(training_data) * min_frequency)))
    # Creating a container to hold each of the edited columns of each data split. During transformation each column
    # of the data becomes a list of entries (one for each row). The outer list holds all the columns in order.
    new_data_split: dict[str, list[list[str]]] = {key: [] for key in data_splits}

    # Run through each of the columns in the training data
    for column_idx in range(training_data.shape[1]):
        counter = Counter(training_data[:, column_idx].tolist())
        popular_categories = {k for k, v in counter.items() if v >= min_count}

        for split, data_split in data_splits.items():
            data_split_column: list[str] = data_split[:, column_idx].tolist()
            collapsed_categories = [
                (cat if cat in popular_categories else CAT_RARE_VALUE) for cat in data_split_column
            ]
            new_data_split[split].append(collapsed_categories)

    return {k: np.array(v).T for k, v in new_data_split.items()}


def encode_categorical_features(
    x: ArrayDict,
    encoding: CategoricalEncoding | None,
    y_train: np.ndarray | None,
    seed: int | None,
    return_encoder: bool = False,
) -> tuple[ArrayDict, bool, Any | None]:
    """
    Encode the categorical features of the dataset.

    Args:
        x: The data to encode.
        encoding: The encoding to use. If None, will use CatEncoding.ORDINAL.
        y_train: The target values. Optional, default is None. Will only be used for the "counter" encoding.
        seed: The seed to use for the random state. Optional, default is None.
        return_encoder: Whether to return the encoder. Optional, default is False.

    Returns:
        A tuple with the following values:
            - The encoded data.
            - A boolean value indicating if the data was converted to numerical.
            - The encoder, if return_encoder is True. None otherwise.
    """
    if encoding != CategoricalEncoding.COUNTER:
        y_train = None

    # Step 1. Map strings to 0-based ranges

    if encoding is None or encoding == CategoricalEncoding.ORDINAL:
        unknown_value = np.iinfo("int64").max - 3
        oe = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=unknown_value,
            dtype="int64",
        ).fit(x[DataSplit.TRAIN.value])
        encoder = make_pipeline(oe)
        encoder.fit(x[DataSplit.TRAIN.value])
        x = {k: encoder.transform(v) for k, v in x.items()}
        max_values = x[DataSplit.TRAIN.value].max(axis=0)
        for part in x:
            if part == DataSplit.TRAIN.value:
                continue
            for column_idx in range(x[part].shape[1]):
                x[part][x[part][:, column_idx] == unknown_value, column_idx] = max_values[column_idx] + 1
        if return_encoder:
            return x, False, encoder
        return x, False, None

    # Step 2. Encode.

    if encoding == CategoricalEncoding.ONE_HOT:
        ohe = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            dtype=np.float32,
        )
        encoder = make_pipeline(ohe)
        encoder.fit(x[DataSplit.TRAIN.value])
        x = {k: encoder.transform(v) for k, v in x.items()}

    elif encoding == CategoricalEncoding.COUNTER:
        assert y_train is not None
        assert seed is not None
        loe = LeaveOneOutEncoder(sigma=0.1, random_state=seed, return_df=False)
        encoder.steps.append(("loe", loe))
        encoder.fit(x[DataSplit.TRAIN.value], y_train)
        x = {k: encoder.transform(v).astype("float32") for k, v in x.items()}
        if not isinstance(x[DataSplit.TRAIN.value], pd.DataFrame):
            x = {k: v.value if hasattr(v, "value") else v for k, v in x.items()}
    else:
        raise ValueError(f"Unsupported encoding: {encoding.value}")

    if return_encoder:
        return x, True, encoder
    return x, True, None


def build_target(y: ArrayDict, policy: TargetPolicy | None, task_type: TaskType) -> tuple[ArrayDict, dict[str, Any]]:
    """
    Build the target and return the target values metadata.

    Args:
        y: The target values.
        policy: The policy to use to build the target. Can be YPolicy.DEFAULT. If none, it will no-op.
        task_type: The type of the task.

    Returns:
        A tuple with the target values and the target values metadata.
    """
    info: dict[str, Any] = {"policy": policy}
    if policy is None:
        pass
    elif policy == TargetPolicy.DEFAULT:
        if task_type == TaskType.REGRESSION:
            mean = float(y[DataSplit.TRAIN.value].mean())
            std = float(y[DataSplit.TRAIN.value].std())
            y = {k: (v - mean) / std for k, v in y.items()}
            info["mean"] = mean
            info["std"] = std
    else:
        raise ValueError(f"Unsupported policy: {policy.value}")

    return y, info
