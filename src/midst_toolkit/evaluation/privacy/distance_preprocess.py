from typing import Any, overload

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from midst_toolkit.evaluation.utils import extract_columns_based_on_meta_info


@overload
def preprocess(
    meta_info: dict[str, Any], synthetic_data: pd.DataFrame, real_data_train: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]: ...


@overload
def preprocess(
    meta_info: dict[str, Any],
    synthetic_data: pd.DataFrame,
    real_data_train: pd.DataFrame,
    real_data_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: ...


def preprocess(
    meta_info: dict[str, Any],
    synthetic_data: pd.DataFrame,
    real_data_train: pd.DataFrame,
    real_data_test: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function performs preprocessing on Pandas dataframes to prepare for computation of various record-to-record
    distances. This is used for computations like distance to closest record scores. Specifically, this function
    filters the provided raw dataframes to the appropriate numerical and categorical columns based on the information
    of the ``meta_info`` JSON. For the numerical columns, it normalizes values by the distance between the largest
    and smallest value of each column of the ``real_data_train`` numerical values. The categorical columns are
    processed into one-hot encoding columns, where the transformation is fitted on the concatenation of columns from
    each dataset.

    Args:
        meta_info: JSON with meta information about the columns and their corresponding types that should be
            considered.
        synthetic_data: Dataframe containing all synthetically generated data.
        real_data_train: Dataframe containing the real training data associated with the model that generated the
            ``synthetic_data``.
        real_data_test: Dataframe containing the real test data. It's important that this data was not seen by the
            model that generated ``synthetic_data`` during training. If None, then it will, of course, not be
            preprocessed. Defaults to None.

    Returns:
        Processed Pandas dataframes with the synthetic data, real data for training, real data for testing if it was
        provided.
    """
    numerical_synthetic_data, categorical_synthetic_data = extract_columns_based_on_meta_info(
        synthetic_data, meta_info
    )
    numerical_real_data_train, categorical_real_data_train = extract_columns_based_on_meta_info(
        real_data_train, meta_info
    )

    numerical_ranges = [
        numerical_real_data_train[index].max() - numerical_real_data_train[index].min()
        for index in numerical_real_data_train.columns
    ]
    numerical_ranges_np = np.array(numerical_ranges)

    num_synthetic_data_np = numerical_synthetic_data.to_numpy()
    num_real_data_train_np = numerical_real_data_train.to_numpy()

    # Normalize the values of the numerical columns of the different datasets by the ranges of the train set.
    num_synthetic_data_np = num_synthetic_data_np / numerical_ranges_np
    num_real_data_train_np = num_real_data_train_np / numerical_ranges_np

    cat_synthetic_data_np = categorical_synthetic_data.to_numpy().astype("str")
    cat_real_data_train_np = categorical_real_data_train.to_numpy().astype("str")

    if real_data_test is not None:
        numerical_real_data_test, categorical_real_data_test = extract_columns_based_on_meta_info(
            real_data_test, meta_info
        )
        num_real_data_test_np = numerical_real_data_test.to_numpy()
        # Normalize the values of the numerical columns of the different datasets by the ranges of the train set.
        num_real_data_test_np = num_real_data_test_np / numerical_ranges_np
        cat_real_data_test_np = categorical_real_data_test.to_numpy().astype("str")
    else:
        num_real_data_test_np, cat_real_data_test_np = None, None

    if categorical_real_data_train.shape[1] > 0:
        encoder = OneHotEncoder()
        if cat_real_data_test_np is not None:
            encoder.fit(np.concatenate((cat_synthetic_data_np, cat_real_data_train_np, cat_real_data_test_np), axis=0))
        else:
            encoder.fit(np.concatenate((cat_synthetic_data_np, cat_real_data_train_np), axis=0))

        cat_synthetic_data_oh = encoder.transform(cat_synthetic_data_np).toarray()
        cat_real_data_train_oh = encoder.transform(cat_real_data_train_np).toarray()
        if cat_real_data_test_np is not None:
            cat_real_data_test_oh = encoder.transform(cat_real_data_test_np).toarray()

    else:
        cat_synthetic_data_oh = np.empty((categorical_synthetic_data.shape[0], 0))
        cat_real_data_train_oh = np.empty((categorical_real_data_train.shape[0], 0))
        if categorical_real_data_test is not None:
            cat_real_data_test_oh = np.empty((categorical_real_data_test.shape[0], 0))

    processed_real_data_train = pd.DataFrame(
        np.concatenate((num_real_data_train_np, cat_real_data_train_oh), axis=1)
    ).astype(float)
    processed_synthetic_data = pd.DataFrame(
        np.concatenate((num_synthetic_data_np, cat_synthetic_data_oh), axis=1)
    ).astype(float)

    if real_data_test is None:
        return (processed_synthetic_data, processed_real_data_train)

    assert num_real_data_test_np is not None
    assert cat_real_data_test_oh is not None
    return (
        processed_synthetic_data,
        processed_real_data_train,
        pd.DataFrame(np.concatenate((num_real_data_test_np, cat_real_data_test_oh), axis=1)).astype(float),
    )
