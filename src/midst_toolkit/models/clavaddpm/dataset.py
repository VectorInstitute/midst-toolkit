"""Defines the dataset functions for the ClavaDDPM model."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import astuple, dataclass, replace
from logging import INFO
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)

from midst_toolkit.common.enumerations import DataSplit, PredictionType, TaskType
from midst_toolkit.common.logger import log
from midst_toolkit.models.clavaddpm.dataset_transformations import (
    collapse_rare_categories,
    drop_rows_according_to_mask,
    encode_categorical_features,
    normalize,
    process_nans_in_categorical_features,
    transform_targets,
)
from midst_toolkit.models.clavaddpm.dataset_utils import (
    dump_pickle,
    encode_and_merge_features,
    get_categorical_and_numerical_column_names,
    get_category_sizes,
    load_pickle,
)
from midst_toolkit.models.clavaddpm.enumerations import (
    ArrayDict,
    CategoricalEncoding,
    CategoricalNaNPolicy,
    IsTargetConditioned,
    Normalization,
    NumericalNaNPolicy,
    TargetPolicy,
)
from midst_toolkit.models.clavaddpm.metrics import calculate_metrics


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
        datasets: ArrayDict = {}
        for split in splits:
            dataset = np.load(directory / f"{dataset_name}_{split}.npy", allow_pickle=True)
            assert isinstance(dataset, np.ndarray), "Dataset must be of type Numpy Array"
            datasets[split] = dataset
        return datasets

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

    @staticmethod
    def make_dataset_from_df(
        data: pd.DataFrame,
        transformations: Transformations,
        is_target_conditioned: IsTargetConditioned,
        info: dict[str, Any],
        data_split_percentages: list[float] | None = None,
        noise_scale: float = 0,
        data_split_random_state: int = 42,
    ) -> tuple[Dataset, dict[int, LabelEncoder], list[str]]:
        """
        Generate a dataset from a pandas DataFrame.

        NOTE: For now, n_classes (which is part of the info dictionary) has to be set to 0. This is because our
        matrix is the concatenation of (x_num, x_cat). In this case, if we have
        is_y_cond == IsTargetConditioned.CONCAT, we can guarantee that y is the first column of the matrix.  However,
        if we have n_classes > 0, then y is not the first column of the matrix.

        Args:
            data: The pandas DataFrame from which to generate the dataset.
            transformations: The transformations to apply to the dataset AFTER creation.
            is_target_conditioned: The condition on the y column.
                IsTargetConditioned.CONCAT: y is concatenated to X, the model learns a joint distribution of (y, X)
                IsTargetConditioned.EMBEDDING: y is not concatenated to X. During computations, y is embedded
                    and added to the latent vector of X
                IsTargetConditioned.NONE: y column is completely ignored

                How does is_target_conditioned affect the generation of y?
                IsTargetConditioned.CONCAT: the model synthesizes (y, X) directly, so y is just the first column
                IsTargetConditioned.EMBEDDING: y is first sampled using empirical distribution of y. The model only
                    synthesizes X. When returning the generated data, we return the generated X and the sampled y.
                    (y is sampled from empirical distribution, instead of being generated by the model). Note that in
                    this way, y is still not independent of X, because the model has been adding the embedding of y
                    to the latent vector of X during computations.
                IsTargetConditioned.NONE: y is synthesized using y's empirical distribution. X is generated by the
                    model. In this case, y is completely independent of X.

            info: A dictionary with metadata about the DataFrame.
            data_split_percentages: The percentages of the dataset to go into train, val, and test splits. The sum of
                the percentages must amount to 1 (within a tolerance of 0.01). Optional, default is [0.7, 0.2, 0.1].
            noise_scale: The scale of the noise to add to the categorical features. Optional, default is 0.
            data_split_random_state: The random state to use for the data split. Will be passed down to the
                ``train_test_split`` function from sklearn. Optional, default is 42.

        Returns:
            A tuple with:
            - The dataset object containing the created dataset,
            - The label encoders for the categorical columns as a dictionary mapping column INDEX within the
              categorical columns to a label encoder for that column.
            - The column names, with numerical columns first, then categorical columns. Within these two categories,
              column names are in the order they appear in the dataset.
        """
        if data_split_percentages is None:
            data_split_percentages = [0.7, 0.2, 0.1]

        assert len(data_split_percentages) == 3, "The ratios must be a list of 3 values (train, validation, test)."
        assert np.isclose(sum(data_split_percentages), 1, atol=0.01), (
            "The sum of the ratios must amount to 1 (with a tolerance of 0.01)."
        )

        train_percent, validation_percent, test_percent = data_split_percentages
        train_val_data, test_data = train_test_split(
            data,
            test_size=test_percent,
            random_state=data_split_random_state,
        )
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=validation_percent / (train_percent + validation_percent),
            random_state=data_split_random_state,
        )

        categorical_column_names, numerical_column_names = get_categorical_and_numerical_column_names(
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

        column_orders = numerical_column_names + categorical_column_names

        # Encode the categorical features and merge them with the numerical features
        features, label_encoders = encode_and_merge_features(
            categorical_features,
            numerical_features,
            noise_scale,
        )

        assert isinstance(info["n_classes"], int)

        dataset = Dataset(
            features,
            None,
            target,
            y_info={},
            task_type=TaskType(info["task_type"]),
            n_classes=info["n_classes"],
        )

        return transform_dataset(dataset, transformations, None), label_encoders, column_orders


def setup_cache_path(transformations: Transformations, cache_dir: Path | None) -> Path | None:
    """
    Setup the cache path for the transformations and transformed dataset. This will be used to check if a cache for
    the specified transformations already exists. If they don't already exist, this is where they will be saved.

    Args:
        transformations: Set of transformations to be cached.
        cache_dir: Directory to look for the cached transformations and datasets pickle. This will be used as the
            stub and the path will be determined by the specified transformations

    Returns:
        _description_
    """
    if cache_dir is None:
        log(INFO, "No cache_dir provided. Will not attempt to load or save transformed dataset from/to cache")
        return None
    transformations_md5 = hashlib.md5(str(transformations).encode("utf-8")).hexdigest()
    transformations_str = "__".join(map(str, astuple(transformations)))
    return cache_dir / f"cache__{transformations_str}__{transformations_md5}.pickle"


def get_cached_dataset(cache_path: Path, transformations: Transformations) -> Dataset:
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


def transform_dataset(
    dataset: Dataset,
    transformations: Transformations,
    cache_dir: Path | None,
) -> Dataset:
    """
    Fits and applies the given set of transformations to the contents of the provided dataset and returns the
    transformed dataset. If an appropriate cache is specified and exists, this function simply loads an already
    transformed dataset from the cache. If a cache does not exist, this function will cache the dataset and
    transformations there in addition to returning the transformed dataset.

    Args:
        dataset: The dataset to transform.
        transformations: The transformations to apply to the dataset.
        cache_dir: The directory to cache the transformed dataset. Optional, default is None. If not None, will check
            if the transformations and dataset exist in the cache directory. If they do, will returned the cached
            transformed dataset. If not, will transform the dataset and cache it.

    Returns:
        The transformed dataset.
    """
    cache_path = setup_cache_path(transformations, cache_dir)
    if cache_path is not None and cache_path.exists():
        return get_cached_dataset(cache_path, transformations)

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

    if categorical_features is not None:
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

    target, target_info = transform_targets(dataset.y, transformations.target_policy, dataset.task_type)

    dataset = replace(dataset, x_num=numerical_features, x_cat=categorical_features, y=target, y_info=target_info)
    dataset.numerical_transform = numerical_transform
    dataset.categorical_transform = categorical_transform

    if cache_path is not None:
        dump_pickle((transformations, dataset), cache_path)

    return dataset


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
