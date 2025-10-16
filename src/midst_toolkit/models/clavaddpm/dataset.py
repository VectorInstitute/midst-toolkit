"""Defines the dataset functions for the ClavaDDPM model."""

import hashlib
import json
import pickle
from collections import Counter
from copy import deepcopy
from dataclasses import astuple, dataclass, replace
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
from midst_toolkit.models.clavaddpm.enumerations import (
    ArrayDict,
    CategoricalEncoding,
    CategoricalNaNPolicy,
    IsTargetCondioned,
    Normalization,
    NumericalNaNPolicy,
    TargetPolicy,
)


# TODO: Dunders are special case in python, rename these values to something else.
CAT_MISSING_VALUE = "__nan__"
CAT_RARE_VALUE = "__rare__"


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
        # if cache_dir is not None, will save the cahe file path into the cache_path variable
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
    x_num = dataset.x_num

    if x_num is not None and transformations.normalization is not None:
        x_num, numerical_transform = normalize(  # type: ignore[assignment]
            x_num,
            transformations.normalization,
            transformations.seed,
            return_normalizer=True,
        )

    if dataset.x_cat is None:
        assert transformations.categorical_nan_policy is None
        assert transformations.category_minimum_frequency is None
        # assert transformations.cat_encoding is None
        x_cat = None
    else:
        x_cat = process_nans_in_categorical_features(dataset.x_cat, transformations.categorical_nan_policy)
        if transformations.category_minimum_frequency is not None:
            x_cat = drop_rare_categories(x_cat, transformations.category_minimum_frequency)
        x_cat, is_num, categorical_transform = encode_categorical_features(
            x_cat,
            transformations.categorical_encoding,
            dataset.y[DataSplit.TRAIN.value],
            transformations.seed,
            return_encoder=True,
        )
        if is_num:
            x_num = x_cat if x_num is None else {x: np.hstack([x_num[x], x_cat[x]]) for x in x_num}
            x_cat = None

    y, y_info = build_target(dataset.y, transformations.target_policy, dataset.task_type)

    dataset = replace(dataset, x_num=x_num, x_cat=x_cat, y=y, y_info=y_info)
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
# TODO: fix this hideous output type
def normalize(
    x: ArrayDict,
    normalization: Normalization,
    seed: int | None,
    return_normalizer: bool = False,
) -> ArrayDict | tuple[ArrayDict, StandardScaler | MinMaxScaler | QuantileTransformer]:
    """
    Normalize the input data.

    Args:
        x: The data to normalize.
        normalization: The normalization to use.
        seed: The seed to use for the random state. Optional, default is None.
        return_normalizer: Whether to return the normalizer. Optional, default is False.

    Returns:
        The normalized data. If return_normalizer is True, will return a tuple with the
            normalized data and the normalizer.
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
    if return_normalizer:
        return {k: normalizer.transform(v) for k, v in x.items()}, normalizer
    return {k: normalizer.transform(v) for k, v in x.items()}


# TODO: is there any relationship between this function and the cat_process_nans function?
# Can they be made a little more similar to each other (in terms of signature)?
def process_nans_in_numerical_features(dataset: Dataset, policy: NumericalNaNPolicy | None) -> Dataset:
    """
    Process the NaN values in the numerical features of the dataset.

    Args:
        dataset: The dataset to process.
        policy: The policy to use to process the NaN values.

    Returns:
        The processed dataset.
    """
    assert dataset.x_num is not None
    nan_masks = {k: np.isnan(v) for k, v in dataset.x_num.items()}
    if not any(mask.any() for mask in nan_masks.values()):
        assert policy is None
        return dataset

    assert policy is not None
    if policy == NumericalNaNPolicy.DROP_ROWS:
        valid_masks = {k: ~v.any(1) for k, v in nan_masks.items()}
        assert valid_masks[DataSplit.TEST.value].all(), (
            "Cannot drop test rows, since this will affect the final metrics."
        )
        new_data = {}
        for data_name in ["x_num", "x_cat", "y"]:
            # TODO: find a way to do this without getattr
            data_dict = getattr(dataset, data_name)
            if data_dict is not None:
                new_data[data_name] = {k: v[valid_masks[k]] for k, v in data_dict.items()}
        dataset = replace(dataset, **new_data)  # type: ignore[arg-type]
    elif policy == NumericalNaNPolicy.MEAN:
        new_values = np.nanmean(dataset.x_num[DataSplit.TRAIN.value], axis=0)  # type: ignore[index]
        x_num = deepcopy(dataset.x_num)
        for k, v in x_num.items():  # type: ignore[union-attr]
            num_nan_indices = np.where(nan_masks[k])
            v[num_nan_indices] = np.take(new_values, num_nan_indices[1])
        dataset = replace(dataset, x_num=x_num)
    else:
        raise ValueError(f"Unsupported policy: {policy.value}")
    return dataset


def process_nans_in_categorical_features(x: ArrayDict, policy: CategoricalNaNPolicy | None) -> ArrayDict:
    """
    Process the NaN values in the categorical features of the dataset.

    Args:
        x: The data to process.
        policy: The policy to use to process the NaN values. If none, will no-op.

    Returns:
        The processed data.
    """
    assert x is not None
    nan_masks = {k: v == CAT_MISSING_VALUE for k, v in x.items()}
    if any(mask.any() for mask in nan_masks.values()):
        if policy is None:
            x_new = x
        elif policy == CategoricalNaNPolicy.MOST_FREQUENT:
            imputer = SimpleImputer(missing_values=CAT_MISSING_VALUE, strategy=policy)
            imputer.fit(x[DataSplit.TRAIN.value])
            x_new = {k: cast(np.ndarray, imputer.transform(v)) for k, v in x.items()}
        else:
            raise ValueError(f"Unsupported cat_nan_policy: {policy.value}")
    else:
        assert policy is None
        x_new = x
    return x_new


def drop_rare_categories(x: ArrayDict, min_frequency: float) -> ArrayDict:
    """
    Drop the rare categories in the categorical data.

    Args:
        x: The data to drop the rare categories from.
        min_frequency: The minimum frequency threshold of the categories to keep. Has to be between 0 and 1.

    Returns:
        The processed data.
    """
    assert 0.0 < min_frequency < 1.0, "min_frequency has to be between 0 and 1"
    min_count = round(len(x[DataSplit.TRAIN.value]) * min_frequency)
    x_new: dict[str, list[Any]] = {key: [] for key in x}
    for column_idx in range(x[DataSplit.TRAIN.value].shape[1]):
        counter = Counter(x[DataSplit.TRAIN.value][:, column_idx].tolist())
        popular_categories = {k for k, v in counter.items() if v >= min_count}
        for part, _ in x_new.items():
            x_new[part].append(
                [(cat if cat in popular_categories else CAT_RARE_VALUE) for cat in x[part][:, column_idx].tolist()]
            )
    return {k: np.array(v).T for k, v in x_new.items()}


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

        # encoder.steps.append(('ohe', ohe))
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
            x = {k: v.values for k, v in x.items()}  # type: ignore[attr-defined]
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
