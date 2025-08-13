import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


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
        LOGGER.warning(f"Path: {save_directory} already exists. Make sure this is intended")


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
        LOGGER.warning(f"File at path {file_path} already exists.")
    with open(file_path, "w") as f:
        for metric_key, metric_value in metrics_dict.items():
            f.write(f"Metric Name: {metric_key}\t Metric Value: {metric_value}\n")


def extract_columns_based_on_meta_info(data: pd.DataFrame, meta_info: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given a set of meta information, which should be in JSON format with keys 'num_col_idx', 'cat_col_idx',
    'target_col_idx', and 'task_type', the provided dataframe is filtered to the correct set of columns for evaluation
    using the meta information.

    Args:
        data: Dataframe to be filtered using the meta information
        meta_info: JSON with meta information about the columns and their corresponding types that should be
            considered.

    Returns:
        Filtered dataframes. The first
    """
    # Enumerate columns and replace column name with index
    data.columns = range(len(data.columns))

    # Get numerical and categorical column indices from meta info
    # NOTE: numerical and categorical columns are the only admissible/generate-able types"
    numerical_column_idx = meta_info["num_col_idx"]
    categorical_column_idx = meta_info["cat_col_idx"]

    # Target columns are also part of the generation, just need to add it to the right "category"
    target_col_idx = meta_info["target_col_idx"]
    if meta_info["task_type"] == "regression":
        numerical_column_idx += target_col_idx
    else:
        categorical_column_idx += target_col_idx

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
    _summary_.

    Args:
        real_categorical_data: _description_
        synthetic_categorical_data: _description_
        real_numerical_data: _description_
        synthetic_numerical_data: _description_

    Returns:
        _description_
    """
    encoder = OneHotEncoder()
    one_hot_real_data = encoder.fit_transform(real_categorical_data)
    one_hot_synthetic_data = encoder.transform(synthetic_categorical_data)

    real_dataframe = pd.DataFrame(np.concatenate((real_numerical_data, one_hot_real_data), axis=1)).astype(float)

    synthetic_dataframe = pd.DataFrame(
        np.concatenate((synthetic_numerical_data, one_hot_synthetic_data), axis=1)
    ).astype(float)

    return real_dataframe, synthetic_dataframe
