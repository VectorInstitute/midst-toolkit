import os
from logging import WARNING
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from midst_toolkit.common.enumerations import TaskType
from midst_toolkit.common.logger import log


def create_quality_metrics_directory(save_directory: Path) -> None:
    """
    Helper function for creating a directory at the specified path to whole metrics results. If the directory already
    exists, this function will log a warning and no-op.

    Args:
        save_directory: Path of the directory to create.
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    else:
        log(WARNING, f"Path: {save_directory} already exists. Make sure this is intended.")


def dump_metrics_dict(metrics_dict: dict[str, float], file_path: Path) -> None:
    """
    Write the provided metrics dictionary to the provided ``file_path`` argument. The metrics dictionary is written
    in a specific format.

    Args:
        metrics_dict: Dictionary of metrics with string key values and associated floats representing metrics
            calculations
        file_path: Path to which the metrics are written. The file will be created or overwritten if it exists
    """
    if os.path.exists(file_path):
        log(WARNING, f"File at path {file_path} already exists.")
    with open(file_path, "w") as f:
        for metric_key, metric_value in metrics_dict.items():
            f.write(f"Metric Name: {metric_key}\t Metric Value: {metric_value}\n")


def extract_columns_based_on_meta_info(
    data: pd.DataFrame, meta_info: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given a set of meta information, which should be in JSON format with keys 'num_col_idx', 'cat_col_idx',
    'target_col_idx', and 'task_type', the provided dataframe is filtered to the correct set of columns for evaluation
    using the meta information.

    Args:
        data: Dataframe to be filtered using the meta information
        meta_info: JSON with meta information about the columns and their corresponding types that should be
            considered. At minimum, it should have the keys 'num_col_idx', 'cat_col_idx', 'target_col_idx', and
            'task_type'

    Returns:
        Filtered dataframes. The first dataframe is the filtered set of columns associated with numerical data. The
        second is the filtered set of columns associated with categorical data.
    """
    # TODO: Consider creating a meta_info class that formalizes the structure of the meta_info produced when
    # Training the diffusion generators.

    # Enumerate columns and replace column name with index
    data.columns = range(len(data.columns))

    # Get numerical and categorical column indices from meta info
    # NOTE: numerical and categorical columns are the only admissible/generate-able types"
    numerical_column_idx = meta_info["num_col_idx"]
    categorical_column_idx = meta_info["cat_col_idx"]

    # Target columns are also part of the generation, just need to add it to the right "category"
    target_col_idx = meta_info["target_col_idx"]
    task_type = TaskType(meta_info["task_type"])
    if task_type == TaskType.REGRESSION:
        numerical_column_idx = numerical_column_idx + target_col_idx
    else:
        categorical_column_idx = categorical_column_idx + target_col_idx

    numerical_data = data[numerical_column_idx]
    categorical_data = data[categorical_column_idx]

    return numerical_data, categorical_data


def one_hot_encode_categoricals_and_merge_with_numerical(
    real_categorical_data: np.ndarray,
    synthetic_categorical_data: np.ndarray,
    real_numerical_data: np.ndarray,
    synthetic_numerical_data: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs one-hot encoding on the real and synthetic data contained in numpy arrays. The ``real_categorical_data``
    is used to fit the one-hot encoder, which is then applied to the data in ``synthetic_categorical_data``. The
    resulting, one-hot encoded, numpy arrays are then concatenated together numerical then one-hots for both the
    synthetic and real data.

    Args:
        real_categorical_data: Categorical data from the real dataset.
        synthetic_categorical_data: Categorical data from the synthetically generated dataset.
        real_numerical_data: Numerical data from the real dataset.
        synthetic_numerical_data: Numerical data from the synthetically generated dataset.

    Returns:
        Two pandas dataframes representing the numerical and categorical data concatenated together. First dataframe
        is the real data, second is the synthetic data.
    """
    encoder = OneHotEncoder()
    one_hot_real_data = encoder.fit_transform(real_categorical_data).toarray()
    one_hot_synthetic_data = encoder.transform(synthetic_categorical_data).toarray()

    real_dataframe = pd.DataFrame(np.concatenate((real_numerical_data, one_hot_real_data), axis=1)).astype(float)

    synthetic_dataframe = pd.DataFrame(
        np.concatenate((synthetic_numerical_data, one_hot_synthetic_data), axis=1)
    ).astype(float)

    return real_dataframe, synthetic_dataframe
