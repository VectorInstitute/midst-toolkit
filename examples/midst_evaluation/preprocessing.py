from typing import Any, overload

import pandas as pd

from midst_toolkit.common.enumerations import TaskType
from midst_toolkit.data_processing.utils import SynthEvalDataframeEncoding
from midst_toolkit.evaluation.utils import (
    extract_columns_based_on_meta_info,
    one_hot_encode_categoricals_and_merge_with_numerical,
)


def preprocess_data_for_alpha_precision_eval(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame, meta_info: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function used to apply specific dataset preprocessing steps related to performing Alpha (and Beta) precision
    measurements comparing ``real_data`` to ``synthetic_data``. Specifically, categorical columns are one-hot encoded
    and numerical columns remain unchanged.

    This follows the convention of:

    https://github.com/VectorInstitute/MIDSTModels/blob/main/midst_models/single_table_TabDDPM/eval/eval_quality.py

    Args:
        real_data: Dataframe containing real data to which the synthetic data will be compared.
        synthetic_data: Dataframe containing the synthetic data whose quality is to be measured.
        meta_info: Dictionary containing meta information. This is used to find the columns in the dataframes
            associated with numerical and categorical data.

    Returns:
        A tuple of preprocessed versions of the real and synthetic dataframes, in that order.
    """
    numerical_real_data, categorical_real_data = extract_columns_based_on_meta_info(real_data, meta_info)
    numerical_synthetic_data, categorical_synthetic_data = extract_columns_based_on_meta_info(
        synthetic_data, meta_info
    )

    numerical_real_numpy, categorical_real_numpy, numerical_synthetic_numpy, categorical_synthetic_numpy = (
        numerical_real_data.to_numpy(),
        categorical_real_data.to_numpy().astype("str"),
        numerical_synthetic_data.to_numpy(),
        categorical_synthetic_data.to_numpy().astype("str"),
    )

    return one_hot_encode_categoricals_and_merge_with_numerical(
        categorical_real_numpy, categorical_synthetic_numpy, numerical_real_numpy, numerical_synthetic_numpy
    )


def get_numerical_and_categorical_column_names(
    data: pd.DataFrame, meta_info: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """
    Based on the information in ``meta_info`` the names of the numerical and categorical columns of the
    provided dataframe are extracted from ``data`` and returned.

    Args:
        data: Collection of data with a set of column names that will be extracted
        meta_info: Dictionary of metadata, including which column indices correspond to the numerical and categorical
            columns of the provided dataset.

    Returns:
        A tuple of the names of numerical and categorical columns, respectively.
    """
    # Enumerate columns and replace column name with index
    column_names = list(data.columns)

    # Get numerical and categorical column indices from meta info
    # NOTE: numerical and categorical columns are the only admissible/generate-able types"
    numerical_column_idx = meta_info["num_col_idx"]
    categorical_column_idx = meta_info["cat_col_idx"]

    if "target_col_idx" in meta_info:
        # Target columns are also part of the generation, just need to add it to the right "category"
        target_col_idx = meta_info["target_col_idx"]
        task_type = TaskType(meta_info["task_type"])
        if task_type == TaskType.REGRESSION:
            numerical_column_idx = numerical_column_idx + target_col_idx
        else:
            categorical_column_idx = categorical_column_idx + target_col_idx

    return [column_names[i] for i in numerical_column_idx], [column_names[i] for i in categorical_column_idx]


@overload
def syntheval_preprocess(
    numerical_columns: list[str], categorical_columns: list[str], real_data: pd.DataFrame, synthetic_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]: ...


@overload
def syntheval_preprocess(
    numerical_columns: list[str],
    categorical_columns: list[str],
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    holdout_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: ...


def syntheval_preprocess(
    numerical_columns: list[str],
    categorical_columns: list[str],
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    holdout_data: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs preprocessing steps on ``real_data``, ``synthetic_data``, and ``holdout_data`` (if provided) using the
    standard preprocessing pipeline used in the SynthEval library. That is, numerical columns are min-max encoded
    and categorical columns are ordinally encoded (not one-hot encoded). If all three dataframes are provided,
    fitting is performed on all three.

    Args:
        numerical_columns: Numerical column names in the respective dataframes.
        categorical_columns: Categorical column names in the respective dataframes.
        real_data: Dataframe containing real data (often used to train the model that generated ``synthetic_data``)
        synthetic_data: Dataframe containing synthetic data.
        holdout_data: Dataframe containing real data (often explicitly NOT used to train the model that generated
            ``synthetic_data``). If None, then fitting and preprocessing is based only on ``real_data`` and
            ``synthetic_data``. Defaults to None.

    Returns:
        A tuple containing the preprocessed real, synthetic, and possibly holdout dataframes, in that order.
    """
    encoder = SynthEvalDataframeEncoding(
        real_data, synthetic_data, categorical_columns, numerical_columns, holdout_data=holdout_data
    )
    real_data = encoder.encode(real_data)
    synthetic_data = encoder.encode(synthetic_data)

    if holdout_data is not None:
        return real_data, synthetic_data, encoder.encode(holdout_data)
    return real_data, synthetic_data
