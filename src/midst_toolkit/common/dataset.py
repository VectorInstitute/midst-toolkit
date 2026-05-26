"""Defines the dataset functions that are common to all models."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import astuple, dataclass
from logging import INFO
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from midst_toolkit.common.dataset_utils import (
    get_category_sizes,
    load_pickle,
)
from midst_toolkit.common.enumerations import (
    ArrayDict,
    CategoricalEncoding,
    CategoricalNaNPolicy,
    DataSplit,
    IsTargetConditioned,
    Normalization,
    NumericalNaNPolicy,
    PredictionType,
    TargetPolicy,
    TaskType,
)
from midst_toolkit.common.logger import log
from midst_toolkit.common.metrics import calculate_metrics


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
    def default(cls) -> Transformations:
        """Return the default transformations."""
        return cls(seed=0, normalization=Normalization.QUANTILE, target_policy=TargetPolicy.DEFAULT)

    @classmethod
    def from_dict(cls, transformations_dict: dict[str, Any]) -> Transformations:
        """Create a Transformations object from a dictionary.

        Args:
            transformations_dict: The dictionary to create the Transformations object from.

        Returns:
            The Transformations object.
        """
        parsed_dict = deepcopy(transformations_dict)

        if "seed" in parsed_dict:
            if parsed_dict["seed"] is None:
                raise ValueError("Seed cannot be None")
            parsed_dict["seed"] = int(parsed_dict["seed"])

        if "normalization" in parsed_dict and parsed_dict["normalization"] is not None:
            parsed_dict["normalization"] = Normalization(parsed_dict["normalization"])

        if "numerical_nan_policy" in parsed_dict and parsed_dict["numerical_nan_policy"] is not None:
            parsed_dict["numerical_nan_policy"] = NumericalNaNPolicy(parsed_dict["numerical_nan_policy"])

        if "categorical_nan_policy" in parsed_dict and parsed_dict["categorical_nan_policy"] is not None:
            parsed_dict["categorical_nan_policy"] = CategoricalNaNPolicy(parsed_dict["categorical_nan_policy"])

        if "category_minimum_frequency" in parsed_dict and parsed_dict["category_minimum_frequency"] is not None:
            parsed_dict["category_minimum_frequency"] = float(parsed_dict["category_minimum_frequency"])

        if "categorical_encoding" in transformations_dict and transformations_dict["categorical_encoding"] is not None:
            parsed_dict["categorical_encoding"] = CategoricalEncoding(parsed_dict["categorical_encoding"])

        if "target_policy" in parsed_dict and parsed_dict["target_policy"] is not None:
            parsed_dict["target_policy"] = TargetPolicy(parsed_dict["target_policy"])

        return cls(**parsed_dict)


@dataclass(frozen=True)
class TableMetadata:
    categorical_column_names: list[str]
    numerical_column_names: list[str]
    target_column_name: str
    n_classes: int
    task_type: TaskType


@dataclass
class TargetInfo:
    policy: TargetPolicy | None = None
    mean: float | None = None
    std: float | None = None


@dataclass
class Dataset:
    numerical_features: ArrayDict | None
    categorical_features: ArrayDict | None
    target: ArrayDict
    target_info: TargetInfo
    task_type: TaskType
    n_classes: int | None
    categorical_transform: OneHotEncoder | None = None
    numerical_transform: StandardScaler | None = None

    @classmethod
    def from_dir(cls, directory: Path) -> Dataset:
        """
        Load a dataset from a directory.

        Args:
            directory: The directory to load the dataset from. Can be a Path object or a path string.

        Returns:
            The loaded dataset.
        """
        json_path = Path(directory / "info.json")
        assert json_path.exists(), f"Info.json file not found at {json_path}"

        info = json.loads(json_path.read_text())

        return cls(
            cls._load_datasets(directory, "x_num") if directory.joinpath("x_num_train.npy").exists() else None,
            cls._load_datasets(directory, "x_cat") if directory.joinpath("x_cat_train.npy").exists() else None,
            cls._load_datasets(directory, "y"),
            TargetInfo(),
            TaskType(info["task_type"]),
            info.get("n_classes"),
        )

    @classmethod
    def _load_datasets(cls, directory: Path, dataset_name: str) -> ArrayDict:
        """
        Load all the dataset splits from a directory. Will check which of the splits exist in the directory for the
        given ``dataset_name`` and load all of them.

        Args:
            directory: The directory to load the dataset from.
            dataset_name: The dataset_name to load.

        Returns:
            The loaded datasets with all the splits.
        """
        splits = [k.value for k in list(DataSplit) if directory.joinpath(f"y_{k.value}.npy").exists()]
        if not len(splits) > 0:
            raise ValueError("Splits to be loaded is empty!")
        datasets: ArrayDict = {}

        for split in splits:
            dataset = np.load(directory / f"{dataset_name}_{split}.npy", allow_pickle=True)
            assert isinstance(dataset, np.ndarray), "Dataset must be of type Numpy Array"
            datasets[split] = dataset

        return datasets

    @property
    def is_binary_classification(self) -> bool:
        """
        Check if the dataset is a binary classification dataset.

        Returns:
            True if the dataset is a binary classification dataset, False otherwise.
        """
        return self.task_type == TaskType.BINARY_CLASSIFICATION

    @property
    def is_multiclass_classification(self) -> bool:
        """
        Check if the dataset is a multiclass classification dataset.

        Returns:
            True if the dataset is a multiclass classification dataset, False otherwise.
        """
        return self.task_type == TaskType.MULTICLASS_CLASSIFICATION

    @property
    def is_regression(self) -> bool:
        """
        Check if the dataset is a regression dataset.

        Returns:
            True if the dataset is a regression dataset, False otherwise.
        """
        return self.task_type == TaskType.REGRESSION

    @property
    def n_numerical_features(self) -> int:
        """
        Get the number of numerical features in the dataset.

        That number should be in the second dimension of the tensors of x_num.

        Returns:
            The number of numerical features in the dataset.
        """
        return 0 if self.numerical_features is None else self.numerical_features[DataSplit.TRAIN.value].shape[1]

    @property
    def n_categorical_features(self) -> int:
        """
        Get the number of categorical features in the dataset.

        That number should be in the second dimension of the tensors of x_cat.

        Returns:
            The number of categorical features in the dataset.
        """
        return 0 if self.categorical_features is None else self.categorical_features[DataSplit.TRAIN.value].shape[1]

    @property
    def n_features(self) -> int:
        """
        Get the total number of features in the dataset.

        Returns:
            The total number of features in the dataset.
        """
        return self.n_numerical_features + self.n_categorical_features

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
        return sum(map(len, self.target.values())) if split is None else len(self.target[split.value])

    @property
    def output_dimension(self) -> int:
        """
        Get the output dimension of the model.

        For self.task_type == TaskType.MULTICLASS_CLASSIFICATION, the output dimension is the number of classes.
        For self.task_type == TaskType.REGRESSION, the output dimension is 1.
        For self.task_type == TaskType.BINARY_CLASSIFICATION, the output dimension is also 1 because
            it is label encoded.

        Returns:
            The output dimension of the model.
        """
        if self.is_multiclass_classification:
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
        return [] if self.categorical_features is None else get_category_sizes(self.categorical_features[split.value])

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
            x: calculate_metrics(self.target[x], predictions[x], self.task_type, prediction_type, self.target_info.std)
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


# Type generics to make the code flexible to any subclass of Dataset
TDataset = TypeVar("TDataset", bound=Dataset)


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


def setup_cache_path(transformations: Transformations, cache_dir: Path | None) -> Path | None:
    """
    Setup the cache path for the transformations and transformed dataset. This will be used to check if a cache for
    the specified transformations already exists. If they don't already exist, this is where they will be saved.

    Args:
        transformations: Set of transformations to be cached.
        cache_dir: Directory to look for the tuple of cached transformations and dataset pickle. This will be used as
            the stub and the path will be determined by the specified transformations

    Returns:
        A path to the cache file based on the hash of the transformations and their names. It may exist already
        (will be loaded from there if so) or represent the name of the cache to be saved.
    """
    if cache_dir is None:
        log(INFO, "No cache_dir provided. Will not attempt to load or save transformed dataset from/to cache")
        return None

    transformations_md5 = hashlib.md5(str(transformations).encode("utf-8")).hexdigest()
    transformations_str = "__".join(map(str, astuple(transformations)))

    return cache_dir / f"cache__{transformations_str}__{transformations_md5}.pickle"


# Ignoring the type -var error because mypy does not like when type var is
# returned bu not present in the arguments list.
def get_cached_dataset(cache_path: Path, transformations: Transformations) -> TDataset:  # type: ignore[type-var]
    """
    Provided a ``cache_path`` that exists, we load the contents of the pickle, which should be a tuple of
    Transformations followed by a transformed dataset object. We check if the cached transformations match the
    specified transformations. If they don't, then our cache and the transformations requested are misaligned and we
    throw an error.

    Args:
        cache_path: A Path that has already been verified to exist.
        transformations: A set of desired transformations to have been applied to the cached dataset.

    Raises:
        RuntimeError: Throws if the set of transformations desired does not match those in the cache.

    Returns:
        A cached dataset with the transformations requested already applied.
    """
    cache_transformations, transformed_dataset = load_pickle(cache_path)
    if transformations == cache_transformations:
        log(INFO, f"Using cached features: {cache_path}")
        return transformed_dataset

    raise RuntimeError(f"Hash collision for {cache_path}")


def get_categorical_and_numerical_column_names(
    table_metadata: TableMetadata,
    is_target_conditioned: IsTargetConditioned,
) -> tuple[list[str], list[str]]:
    """
    Get the categorical and numerical column names from the info dictionary. It will also consider whether the target
    variable should be considered a categorical column or not. If ``is_target_conditioned`` is
    ``IsTargetConditioned.CONCAT``, then the label column is considered part of the categorical or numerical columns.
    If table_metadata.n_classes > 0, it is deemed a categorical column. If not, then it is deemed a numerical column.

    Args:
        table_metadata: The TableMetadata object containing metadata for a dataset,
            including the names of the categorical and numerical columns.
        is_target_conditioned: The condition on the y column.

    Returns:
        A tuple of lists with the categorical column names, followed by the numerical column names
    """
    numerical_columns = []
    if table_metadata.numerical_column_names is not None:
        numerical_columns = list(table_metadata.numerical_column_names)

    categorical_columns = []
    if table_metadata.categorical_column_names is not None:
        categorical_columns = list(table_metadata.categorical_column_names)

    if is_target_conditioned == IsTargetConditioned.CONCAT:
        if table_metadata.n_classes > 0:
            categorical_columns.append(table_metadata.target_column_name)
        else:
            numerical_columns.append(table_metadata.target_column_name)

    return categorical_columns, numerical_columns


def process_nans_in_numerical_features(dataset: TDataset, policy: NumericalNaNPolicy | None) -> TDataset:
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

    assert dataset.numerical_features is not None, "No numerical features are present to process."

    nan_masks = {k: np.isnan(v) for k, v in dataset.numerical_features.items()}
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

        dataset.numerical_features = (
            None
            if dataset.numerical_features is None
            else drop_rows_according_to_mask(dataset.numerical_features, valid_masks)
        )
        dataset.categorical_features = (
            None
            if dataset.categorical_features is None
            else drop_rows_according_to_mask(dataset.categorical_features, valid_masks)
        )
        dataset.target = drop_rows_according_to_mask(dataset.target, valid_masks)

    elif policy == NumericalNaNPolicy.MEAN:
        # Computes column means in the training dataset, ignoring NaN values.
        new_values = np.nanmean(dataset.numerical_features[DataSplit.TRAIN.value], axis=0)

        # If any training column is all-NaN, np.nanmean returns NaN
        bad_cols = np.isnan(new_values)
        if bad_cols.any():
            raise ValueError("At least one of the columns in the train split are all NaN")

        numerical_features_per_split = deepcopy(dataset.numerical_features)
        for data_split, numerical_features in numerical_features_per_split.items():
            nan_indices = np.where(nan_masks[data_split])
            numerical_features[nan_indices] = np.take(new_values, nan_indices[1])
        dataset.numerical_features = numerical_features_per_split
    else:
        raise ValueError(f"Unsupported policy: {policy.value}")

    return dataset
