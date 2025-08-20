import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONVERSION_DATASETS = {"default", "news"}
MAX_CLIPPING_DATASETS = {"shoppers"}
MIN_MAX_CLIPPING_DATASETS = {"default", "faults", "beijing"}

CLIPPING_MODEL_PREFIX = "codi"
CONVERSION_MODEL_PREFIX = "great"


def process_midst_data_for_quality_evaluation(
    numerical_real_data: pd.DataFrame,
    categorical_real_data: pd.DataFrame,
    numerical_synthetic_data: pd.DataFrame,
    categorical_synthetic_data: pd.DataFrame,
    dataset_name: str,
    model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function handles data preprocessing customized to some of the models and datasets used in the MIDST
    competition. The processing is drawn from
    https://github.com/VectorInstitute/MIDSTModels/blob/main/midst_models/single_table_TabDDPM/eval/eval_quality.py.

    It has special considerations for how the provided dataframes are processed into numpy arrays depending on the
    dataname and model provided in the arguments.

    Args:
        numerical_real_data: Real data with numerical values
        categorical_real_data: Real data with categorical values
        numerical_synthetic_data: Synthetically generated data with numerical values
        categorical_synthetic_data: Synthetically generated data with numerical values
        dataset_name: Name of the dataset to which the real data belongs. The way that the data is processed will
            depend on whether special treatment is required for the specified name.
        model: Model that was used to generate the synthetic data. Specific model names require special postprocessing
            in order for quality evaluation

    Returns:
        A tuple of four Numpy arrays, one for each of the numerical and categorical collections of real and synthetic
        data after processing. The order is numerical and categorical data for the real data, followed by the same
        for the synthetic data.
    """
    categorical_synthetic_numpy = categorical_synthetic_data.to_numpy().astype("str")

    # Perform some special data post-processing for specific datasets and models as specified in the script
    # arguments

    if dataset_name in CONVERSION_DATASETS and model.startswith(CONVERSION_MODEL_PREFIX):
        # If using the default or news dataset and a model postfixed with "codi," need to perform an int cast
        categorical_synthetic_numpy = categorical_synthetic_data.astype("int").to_numpy().astype("str")
    elif model.startswith(CLIPPING_MODEL_PREFIX):
        if dataset_name in MAX_CLIPPING_DATASETS:
            # Column reassignment
            categorical_synthetic_numpy[:, 1] = categorical_synthetic_data[11].astype("int").to_numpy().astype("str")
            categorical_synthetic_numpy[:, 2] = categorical_synthetic_data[12].astype("int").to_numpy().astype("str")
            categorical_synthetic_numpy[:, 3] = categorical_synthetic_data[13].astype("int").to_numpy().astype("str")

            # Clip the maximum value to reflect that of the real data
            max_data = categorical_real_data[14].max()
            categorical_synthetic_data.loc[categorical_synthetic_data[14] > max_data, 14] = max_data

            # Perform column reassignment
            categorical_synthetic_numpy[:, 4] = categorical_synthetic_data[14].astype("int").to_numpy().astype("str")
            categorical_synthetic_numpy[:, 4] = categorical_synthetic_data[14].astype("int").to_numpy().astype("str")

        elif dataset_name in MIN_MAX_CLIPPING_DATASETS:
            # Note that columns here are not contiguous, so we enumerate
            columns = categorical_real_data.columns
            for i, col in enumerate(columns):
                if categorical_real_data[col].dtype == "int":
                    max_data = categorical_real_data[col].max()
                    min_data = categorical_real_data[col].min()

                    # Perform clipping based on the real data on both sides (min and max)
                    categorical_synthetic_data.loc[categorical_synthetic_data[col] > max_data, col] = max_data
                    categorical_synthetic_data.loc[categorical_synthetic_data[col] < min_data, col] = min_data

                    categorical_synthetic_numpy[:, i] = (
                        categorical_synthetic_data[col].astype("int").to_numpy().astype("str")
                    )
    return (
        numerical_real_data.to_numpy(),
        categorical_real_data.to_numpy().astype("str"),
        numerical_synthetic_data.to_numpy(),
        categorical_synthetic_numpy,
    )


def load_midst_data(
    real_data_path: Path, synthetic_data_path: Path, meta_info_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Helper function for loading data at the specified paths. These paths are constructed either by the user or with a
    particular set of defaults that were used in the original MIDST competition (see, for example,
    https://github.com/VectorInstitute/MIDSTModels/blob/main/midst_models/single_table_TabDDPM/eval/eval_quality.py).

    Args:
        real_data_path: Path from which to load the real data to which the synthetic data will be compared. This
            should be a CSV file.
        synthetic_data_path: Path from which to load the synthetic data to which the real data will be compared. This
            should be a CSV file.
        meta_info_path: This should be a JSON file containing meta information about the data generation process.
            Specifically, it should contain information about which columns of the real and synthetic data should
            actually be compared. It must contain keys: 'num_col_idx', 'cat_col_idx', 'target_col_idx', and
            'task_type'.

    Returns:
        The loaded real data, synthetic data, and meta information json for further processing.
    """
    real_data = pd.read_csv(real_data_path)
    synthetic_data = pd.read_csv(synthetic_data_path)

    with open(meta_info_path, "r") as f:
        meta_info = json.load(f)

    return real_data, synthetic_data, meta_info
