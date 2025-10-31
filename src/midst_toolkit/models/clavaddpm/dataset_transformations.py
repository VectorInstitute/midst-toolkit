from collections import Counter
from logging import INFO
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

from midst_toolkit.common.enumerations import DataSplit, TaskType
from midst_toolkit.common.logger import log
from midst_toolkit.models.clavaddpm.enumerations import (
    ArrayDict,
    CategoricalEncoding,
    CategoricalNaNPolicy,
    Normalization,
    TargetPolicy,
)


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
        normalizer = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(train_split.shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported normalization: {normalization.value}")

    normalizer.fit(train_split)
    return {k: normalizer.transform(v) for k, v in datasets.items()}, normalizer


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
    datasets: ArrayDict,
    encoding: CategoricalEncoding | None,
    y_train: np.ndarray | None,
    seed: int | None,
    return_encoder: bool = False,
) -> tuple[ArrayDict, bool, Any | None]:
    """
    Encode the categorical features of the dataset splits using the encoding strategy specified in the encoding
    argument.

    Args:
        datasets: The data to encode.
        encoding: The kind of encoding to use. If None, will use CatEncoding.ORDINAL.
        y_train: The target values. Will only be used for the "counter" encoding. Optional
        seed: The seed to use for the random state. Only applied when using ``CategoricalEncoding.COUNTER``. Optional
        return_encoder: Whether to return the encoder. Optional, default is False.

    Returns:
        A tuple with the following values:
            - The encoded data.
            - A boolean value indicating if the data was converted to numerical.
            - The encoder, if ``return_encoder`` is True. None otherwise.
    """
    encoding = CategoricalEncoding.ORDINAL if encoding is None else encoding
    y_train = None if encoding != CategoricalEncoding.COUNTER else y_train

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
        assert y_train is not None
        leave_one_out = LeaveOneOutEncoder(sigma=0.1, random_state=seed, return_df=False)
        encoder = make_pipeline(leave_one_out)
        encoder.fit(train_split, y_train)
        datasets = {k: encoder.transform(v).astype("float32") for k, v in datasets.items()}
    else:
        raise ValueError(f"Unsupported encoding: {encoding.value}")

    if return_encoder:
        return datasets, True, encoder
    return datasets, True, None


def transform_targets(
    target_datasets: ArrayDict, policy: TargetPolicy | None, task_type: TaskType
) -> tuple[ArrayDict, dict[str, Any]]:
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
    info: dict[str, Any] = {"policy": policy}
    if policy is None:
        return target_datasets, info

    if policy == TargetPolicy.DEFAULT:
        if task_type == TaskType.REGRESSION:
            train_split = target_datasets[DataSplit.TRAIN.value]
            mean = float(train_split.mean())
            std = float(train_split.std())
            target_datasets = {split: (target_data - mean) / std for split, target_data in target_datasets.items()}
            info["mean"] = mean
            info["std"] = std
    else:
        raise ValueError(f"Unsupported policy: {policy.value}")

    return target_datasets, info
