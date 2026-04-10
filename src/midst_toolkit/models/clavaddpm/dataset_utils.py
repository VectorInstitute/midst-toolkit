from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from midst_toolkit.common.enumerations import DataSplit
from midst_toolkit.models.clavaddpm.enumerations import ArrayDict


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


def dump_pickle(obj: Any, path: Path | str, **kwargs: Any) -> None:
    """
    Dump an object into a pickle file.

    Args:
        obj: The object to dump.
        path: The path to the pickle file.
        **kwargs: Additional arguments to pass to the pickle.dumps function.
    """
    Path(path).write_bytes(pickle.dumps(obj, **kwargs))


def get_category_sizes(features: torch.Tensor | np.ndarray) -> list[int]:
    """
    Get the size of the categories in the features tensor or array provided by counting the number of
    unique values in each column.

    Args:
        features: The data from which to extract category sizes.

    Returns:
        A list with the category sizes in the data.
    """
    columns_list = features.T.cpu().tolist() if isinstance(features, torch.Tensor) else features.T.tolist()
    return [len(set(column)) for column in columns_list]


def encode_and_merge_features(
    categorical_features: ArrayDict | None,
    numerical_features: ArrayDict | None,
    noise_scale: float,
    categorical_column_names: list[str] | None = None,
    label_encoders_path: str | None = None,
) -> tuple[ArrayDict, dict[int, LabelEncoder]]:
    """
    Merge the categorical with the numerical features for train, validation, and test datasets. Numerical features
    are first, followed by categorical features.

    The categorical features are encoded and then merged with the numerical features. The label encoders used to do
    that are also returned.

    If ``noise_scale`` is greater than 0, noise from a normal distribution with a standard deviation of
    ``noise_scale`` is added to the categorical features.

    Args:
        categorical_features: A dictionary with the categorical features data for train, validation, and test datasets.
            keys are "train", "val", "test" from the DataSplit enumeration
        numerical_features: A dictionary with the numerical features data for train, validation, and test datasets.
            keys are "train", "val", "test" from the DataSplit enumeration
        noise_scale: The scale of the noise to add to the categorical features. Noise is drawn from a normal
            distribution with standard deviation of ``noise_scale``.

    Returns:
        The merged features for train, validation, and test datasets and the label encoders used to do so. The label
        encoders are returned as a dictionary mapping column INDEX within the categorical columns to a label encoder
        for that column.
    """
    # if no categorical features, just return the numerical features
    if categorical_features is None:
        assert numerical_features is not None, "Both categorical and numerical features is empty."
        return numerical_features, {}

    # Otherwise, encode the categorical features
    all_categorical_data = np.vstack(
        (
            categorical_features[DataSplit.TRAIN.value],
            categorical_features[DataSplit.VALIDATION.value],
            categorical_features[DataSplit.TEST.value],
        )
    )

    # Load pre-fitted label encoders from pkl if provided, otherwise fit on current data
    preloaded_encoders: dict[str, LabelEncoder] | None = None
    if label_encoders_path is not None:
        _pkl_path = Path(label_encoders_path)
        if _pkl_path.exists():
            with open(_pkl_path, "rb") as _f:
                preloaded_encoders = pickle.load(_f)

    categorical_data_encoded = []
    label_encoders = {}
    for column in range(all_categorical_data.shape[1]):
        col_name = categorical_column_names[column] if categorical_column_names is not None else None
        if preloaded_encoders is not None and col_name is not None and col_name in preloaded_encoders:
            # Use pre-fitted encoder from full dataset (e.g. 101K rows)
            label_encoder = preloaded_encoders[col_name]
            encoded_labels = label_encoder.transform(all_categorical_data[:, column]).astype(float)
        else:
            # Fallback: fit on current data
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(all_categorical_data[:, column]).astype(float)
        if noise_scale > 0:
            # add noise
            encoded_labels += np.random.normal(0, noise_scale, encoded_labels.shape)
        categorical_data_encoded.append(encoded_labels)
        label_encoders[column] = label_encoder

    categorical_data_transposed = np.vstack(categorical_data_encoded).T

    # Split the encoded data back into the train, validation, and test splits.
    num_train_samples = categorical_features[DataSplit.TRAIN.value].shape[0]
    num_validation_samples = categorical_features[DataSplit.VALIDATION.value].shape[0]

    categorical_features[DataSplit.TRAIN.value] = categorical_data_transposed[:num_train_samples, :]
    categorical_features[DataSplit.VALIDATION.value] = categorical_data_transposed[
        num_train_samples : num_train_samples + num_validation_samples, :
    ]
    categorical_features[DataSplit.TEST.value] = categorical_data_transposed[
        num_train_samples + num_validation_samples :, :
    ]

    # if no numerical features then no need to merge, just return the categorical features
    if numerical_features is None:
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
