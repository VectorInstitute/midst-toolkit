from collections import Counter
from dataclasses import replace
from logging import INFO
from pathlib import Path
from typing import Any

import numpy as np
from category_encoders import LeaveOneOutEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileTransformer,
    StandardScaler,
)

from midst_toolkit.common.dataset import (
    TargetInfo,
    TDataset,
    Transformations,
    get_cached_dataset,
    process_nans_in_numerical_features,
    setup_cache_path,
)
from midst_toolkit.common.dataset_utils import dump_pickle
from midst_toolkit.common.enumerations import (
    ArrayDict,
    CategoricalEncoding,
    CategoricalNaNPolicy,
    DataSplit,
    Normalization,
    TargetPolicy,
    TaskType,
)
from midst_toolkit.common.logger import log


# Wildcard value to which all rare categorical variables are mapped
CAT_RARE_VALUE = "_rare_"
CAT_MISSING_VALUE = "_nan_"


# Inspired by: https://github.com/yandex-research/rtdl/blob/a4c93a32b334ef55d2a0559a4407c8306ffeeaee/lib/data.py#L20
def normalize(
    datasets: ArrayDict,
    normalization: Normalization,
    seed: int | None,
) -> tuple[ArrayDict, StandardScaler | MinMaxScaler | QuantileTransformer]:
    """
    Normalize the input data according to the specified normalization strategy of ``normalization``. Normalization is
    fit on the training split of the datasets passed and then applied to all splits.

    Args:
        datasets: The data to normalize.
        normalization: The normalization to use.
        seed: The seed to use for any random state in the normalization strategy. Currently only applicable to
            QuantileTransformer.

    Returns:
        The normalized data and the fitted normalizer class.
    """
    train_split = datasets[DataSplit.TRAIN.value]

    if normalization == Normalization.STANDARD:
        normalizer = StandardScaler()
    elif normalization == Normalization.MINMAX:
        normalizer = MinMaxScaler()
    elif normalization == Normalization.QUANTILE:
        n_samples = train_split.shape[0]
        n_quantiles = max(min(n_samples // 30, 1000), 10)
        n_quantiles = min(n_quantiles, n_samples)
        normalizer = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_quantiles,
            subsample=int(1e9),
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported normalization: {normalization.value}")

    normalizer.fit(train_split)
    return {k: normalizer.transform(v) for k, v in datasets.items()}, normalizer


def process_nans_in_categorical_features(data_splits: ArrayDict, policy: CategoricalNaNPolicy | None) -> ArrayDict:
    """
    Process the NaN values in the categorical features of the datasets provided. Supports only string or float arrays.

    Args:
        data_splits: A dictionary containing data to process, split into different partitions. One of the keys must
            be DataSplit.TRAIN.value.
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

    # If there are any NaN values, try to apply the policy.
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
        data_splits: A dictionary containing data to process, split into different partitions. One of the keys must be
            DataSplit.TRAIN.value..
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
    datasets: ArrayDict,
    encoding: CategoricalEncoding | None,
    target_train: np.ndarray | None,
    seed: int | None,
    return_encoder: bool = False,
) -> tuple[ArrayDict, bool, Any | None]:
    """
    Encode the categorical features of the dataset splits using the encoding strategy specified in the encoding
    argument.

    Args:
        datasets: The data to encode.
        encoding: The kind of encoding to use. If None, will use CatEncoding.ORDINAL.
        target_train: The target values. Will only be used for the "counter" encoding. Optional
        seed: The seed to use for the random state. Only applied when using ``CategoricalEncoding.COUNTER``. Optional
        return_encoder: Whether to return the encoder. Optional, default is False.

    Returns:
        A tuple with the following values:
            - The encoded data.
            - A boolean value indicating if the data was converted to numerical.
            - The encoder, if ``return_encoder`` is True. None otherwise.
    """
    encoding = CategoricalEncoding.ORDINAL if encoding is None else encoding
    target_train = None if encoding != CategoricalEncoding.COUNTER else target_train

    train_split = datasets[DataSplit.TRAIN.value]

    if encoding is None or encoding == CategoricalEncoding.ORDINAL:
        unknown_value = np.iinfo("int64").max - 3
        ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=unknown_value,
            dtype="int64",
        )
        encoder = make_pipeline(ordinal_encoder)
        encoder.fit(train_split)
        datasets = {k: encoder.transform(v) for k, v in datasets.items()}

        max_values = datasets[DataSplit.TRAIN.value].max(axis=0)
        for split_name, data_split in datasets.items():
            # No corrections for train split
            if split_name == DataSplit.TRAIN.value:
                continue
            for column_idx in range(data_split.shape[1]):
                # Rows that match the unknown value for the column
                unknown_value_rows = data_split[:, column_idx] == unknown_value
                # Make unknown values in split one larger than max value in train
                data_split[unknown_value_rows, column_idx] = max_values[column_idx] + 1

        if return_encoder:
            return datasets, False, encoder
        return datasets, False, None

    if encoding == CategoricalEncoding.ONE_HOT:
        one_hot_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
        encoder = make_pipeline(one_hot_encoder)
        encoder.fit(train_split)
        datasets = {k: encoder.transform(v) for k, v in datasets.items()}

    elif encoding == CategoricalEncoding.COUNTER:
        assert target_train is not None
        leave_one_out = LeaveOneOutEncoder(sigma=0.1, random_state=seed, return_df=False)
        encoder = make_pipeline(leave_one_out)
        encoder.fit(train_split, target_train)
        datasets = {k: encoder.transform(v).astype("float32") for k, v in datasets.items()}
    else:
        raise ValueError(f"Unsupported encoding: {encoding.value}")

    if return_encoder:
        return datasets, True, encoder
    return datasets, True, None


def transform_targets(
    target_datasets: ArrayDict,
    policy: TargetPolicy | None,
    task_type: TaskType,
) -> tuple[ArrayDict, TargetInfo]:
    """
    Applies a transformation to the provided target values across data splits based on the policy specified in
    ``policy``. If no policy is provided or the task type is not Regression, nothing is done. If the policy is
    default and the task_type is regression the targets are centered and normalized using the mean and standard
    deviation of the train targets.

    The info dictionary is meant to store the parameters used in the transformations so that they may be inverted
    later.

    Args:
        target_datasets: The target values across the dataset splits.
        policy: The policy to use to build the target. Can be TargetPolicy.DEFAULT. If none, it will no-op.
        task_type: The type of the task.

    Returns:
        A tuple with the transformed target values across datasets and the metadata that stores information about
        how the transformation was performed.
    """
    target_info = TargetInfo(policy=policy)
    if policy is None:
        return target_datasets, target_info

    if policy == TargetPolicy.DEFAULT:
        if task_type == TaskType.REGRESSION:
            train_split = target_datasets[DataSplit.TRAIN.value]
            mean = float(train_split.mean())
            std = float(train_split.std())
            target_datasets = {split: (target_data - mean) / std for split, target_data in target_datasets.items()}
            target_info.mean = mean
            target_info.std = std
    else:
        raise ValueError(f"Unsupported policy: {policy.value}")

    return target_datasets, target_info


def transform_dataset(
    dataset: TDataset,
    transformations: Transformations,
    cache_dir: Path | None,
) -> TDataset:
    """
    Fits and applies the given set of transformations to the contents of the provided dataset and returns the
    transformed dataset. If an appropriate cache is specified and exists, this function simply loads an already
    transformed dataset from the cache. If a cache does not exist, this function will cache the dataset and
    transformations there in addition to returning the transformed dataset.

    Args:
        dataset: The dataset to transform.
        transformations: The transformations to apply to the dataset.
        cache_dir: The directory to cache the transformed dataset. Optional, default is None. If not None, will check
            if the transformations and dataset exist in the cache directory. If they do, will return the cached
            transformed dataset. If not, will transform the dataset and cache it.

    Returns:
        The transformed dataset.
    """
    cache_path = setup_cache_path(transformations, cache_dir)
    if cache_path is not None and cache_path.exists():
        return get_cached_dataset(type(dataset), cache_path, transformations)

    if dataset.numerical_features is not None:
        # Processing NaNs in numerical features here because we need to
        # drop rows with NaNs in case the policy is DROP_ROWS.
        dataset = process_nans_in_numerical_features(dataset, transformations.numerical_nan_policy)

    numerical_transform = None
    categorical_transform = None
    numerical_features = dataset.numerical_features
    categorical_features = dataset.categorical_features

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
            dataset.target[DataSplit.TRAIN.value],
            transformations.seed,
            return_encoder=True,
        )
        if is_numerical:
            # Will be true if the categorical encoding type is ONE_HOT or COUNTER.
            if numerical_features is None:
                numerical_features = categorical_features
            else:
                numerical_features = {
                    x: np.hstack([numerical_features[x], categorical_features[x]]) for x in numerical_features
                }
            categorical_features = None

    target, target_info = transform_targets(dataset.target, transformations.target_policy, dataset.task_type)

    dataset = replace(
        dataset,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target=target,
        target_info=target_info,
        numerical_transform=numerical_transform,
        categorical_transform=categorical_transform,
    )

    if cache_path is not None:
        dump_pickle((transformations, dataset), cache_path)

    return dataset
