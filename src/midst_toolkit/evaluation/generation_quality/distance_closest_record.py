from enum import Enum
from logging import INFO
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from midst_toolkit.common.logger import log
from midst_toolkit.evaluation.generation_quality.utils import extract_columns_based_on_meta_info


class NormType(Enum):
    L1 = "l1"
    L2 = "l2"


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def compute_l1_distance(synthetic_data: torch.Tensor, real_data: torch.Tensor) -> torch.Tensor:
    """
    Compute the smallest l1 distance between each point in the synthetic data tensor compared to all points in the
    real data tensor.

    Args:
        synthetic_data: Tensor of synthetic data. Assumed to be a 2D tensor with batch size first, followed by
            data dimension.
        real_data: Tensor of synthetic data. Assumed to be a 2D tensor with batch size first, followed by
            data dimension.

    Returns:
        A 1D tensor containing the l1 minimum distances between each data point in the synthetic data and all points in
        the real data. Order will be the same as the synthetic data.
    """
    assert synthetic_data.ndim == 2 and real_data.ndim == 2, "Synthetic and Real data tensors should be 2D"
    assert synthetic_data.shape[1] == real_data.shape[1], "Data dimensions do not match for the provided tensors"
    # For synthetic_data (n_synth, data_dim), and real_data (n_real, data_dim), this subtracts
    # every point in real_data from every point in synthetic_data to create a tensor of shape
    # (n_synth, n_real, data_dim).
    point_differences = synthetic_data[:, None] - real_data
    distances = (point_differences).abs().sum(dim=2)
    # Minimum distance of points in n_synth compared to all other points in real_data.
    min_batch_distances, _ = distances.min(dim=1)
    return min_batch_distances


def compute_l2_distance(synthetic_data: torch.Tensor, real_data: torch.Tensor) -> torch.Tensor:
    """
    Compute the smallest l2 distance between each point in the synthetic data tensor compared to all points in the
    real data tensor.

    Args:
        synthetic_data: Tensor of synthetic data. Assumed to be a 2D tensor with batch size first, followed by
            data dimension.
        real_data: Tensor of synthetic data. Assumed to be a 2D tensor with batch size first, followed by
            data dimension.

    Returns:
        A 1D tensor containing the minimum l2 distances between each data point in the synthetic data and all points in
        the real data. Order will be the same as the synthetic data.
    """
    assert synthetic_data.ndim == 2 and real_data.ndim == 2, "Synthetic and Real data tensors should be 2D"
    assert synthetic_data.shape[1] == real_data.shape[1], "Data dimensions do not match for the provided tensors"
    # For synthetic_data (n_synth, data_dim), and real_data (n_real, data_dim), this subtracts
    # every point in real_data from every point in synthetic_data to create a tensor of shape
    # (n_synth, n_real, data_dim).
    point_differences = synthetic_data[:, None] - real_data
    distances = torch.sqrt(torch.pow(point_differences, 2.0).sum(dim=2))
    # Minimum distance of points in n_synth compared to all other points in real_data.
    min_batch_distances, _ = distances.min(dim=1)
    return min_batch_distances


def minimum_distances(
    synthetic_data: torch.Tensor,
    real_data: torch.Tensor,
    batch_size: int | None = None,
    norm: NormType = NormType.L1,
) -> torch.Tensor:
    """
    Function to calculate minimum distances between each point in the synthetic data to those of the real data
    provided. This can be done in batches if specified. Otherwise, the entire computation is done at once.

    Args:
        synthetic_data: The complete set of synthetic data, stacked as a tensor with shape (n_samples, data dimension).
        real_data: The complete set of real data, stacked as a tensor with shape (n_samples, data dimension).
        batch_size: Size of the batches to facilitate computing the minimum distances, if specified. Defaults to None.
        norm: Which type of norm to use as the distance metric. Defaults to NormType.L1.

    Returns:
        A 1D tensor with the minimum distances. Should be of length n_samples. Order will be the same as
        ``synthetic_data.``
    """
    if batch_size is None:
        # If batch size isn't specified, do it all at once.
        batch_size = synthetic_data.size(0)

    # Create a minimum distance for each synthetic data sample
    min_distances = torch.full((synthetic_data.size(0),), float("inf"), device=synthetic_data.device)
    # Iterate through the real data in batches and compute distances
    for start_index in range(0, real_data.size(0), batch_size):
        end_index = min(start_index + batch_size, real_data.size(0))
        real_data_batch = real_data[start_index:end_index]

        if norm is NormType.L1:
            min_batch_distances = compute_l1_distance(synthetic_data, real_data_batch)
        elif norm is NormType.L2:
            min_batch_distances = compute_l2_distance(synthetic_data, real_data_batch)
        else:
            raise ValueError(f"Unrecognized norm type: {str(norm)}")
        min_distances = torch.minimum(min_distances, min_batch_distances)

    return min_distances


def preprocess_for_distance_to_closest_record_score(
    synthetic_data: pd.DataFrame,
    real_data_train: pd.DataFrame,
    real_data_test: pd.DataFrame,
    meta_info: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    This function performs preprocessing on Pandas dataframes to prepare for computation of the distance to closest
    record score. Specifically, this function filters the provided raw dataframes to the appropriate numerical and
    categorical columns based on the information of the ``meta_info`` JSON. For the numerical columns, it normalizes
    values by the distance between the largest and smallest value of each column of the ``real_data_train`` numerical
    values. The categorical columns are processed into one-hot encoding columns, where the transformation is fitted
    on the concatenation of columns from each dataset.

    Args:
        synthetic_data: Dataframe containing all synthetically generated data.
        real_data_train: Dataframe containing the real training data associated with the model that generated the
            ``synthetic_data``.
        real_data_test: Dataframe containing the real test data. It's important that this data was not seen by the
            model that generated ``synthetic_data`` during training.
        meta_info: JSON with meta information about the columns and their corresponding types that should be
            considered.

    Returns:
        Processed Pandas dataframes the real data for training, real data for testing, and synthetic data.
    """
    numerical_real_data_train, categorical_real_data_train = extract_columns_based_on_meta_info(
        real_data_train, meta_info
    )
    numerical_real_data_test, categorical_real_data_test = extract_columns_based_on_meta_info(
        real_data_test, meta_info
    )
    numerical_synthetic_data, categorical_synthetic_data = extract_columns_based_on_meta_info(
        synthetic_data, meta_info
    )

    numerical_ranges = [
        numerical_real_data_train[index].max() - numerical_real_data_train[index].min()
        for index in numerical_real_data_train.columns
    ]
    numerical_ranges_np = np.array(numerical_ranges)

    num_real_data_train_np = numerical_real_data_train.to_numpy()
    num_real_data_test_np = numerical_real_data_test.to_numpy()
    num_synthetic_data_np = numerical_synthetic_data.to_numpy()

    # Normalize the values of the numerical columns of the different datasets by the ranges of the train set.
    num_real_data_train_np = num_real_data_train_np / numerical_ranges_np
    num_real_data_test_np = num_real_data_test_np / numerical_ranges_np
    num_synthetic_data_np = num_synthetic_data_np / numerical_ranges_np

    cat_real_data_train_np = categorical_real_data_train.to_numpy().astype("str")
    cat_real_data_test_np = categorical_real_data_test.to_numpy().astype("str")
    cat_synthetic_data_np = categorical_synthetic_data.to_numpy().astype("str")

    if categorical_real_data_train.shape[1] > 0:
        encoder = OneHotEncoder()
        encoder.fit(np.concatenate((cat_real_data_train_np, cat_synthetic_data_np, cat_real_data_test_np), axis=0))

        cat_real_data_train_oh = encoder.transform(cat_real_data_train_np).toarray()
        cat_real_data_test_oh = encoder.transform(cat_real_data_test_np).toarray()
        cat_synthetic_data_oh = encoder.transform(cat_synthetic_data_np).toarray()
    else:
        cat_real_data_train_oh = np.empty((categorical_real_data_train.shape[0], 0))
        cat_real_data_test_oh = np.empty((categorical_real_data_test.shape[0], 0))
        cat_synthetic_data_oh = np.empty((categorical_synthetic_data.shape[0], 0))

    return (
        pd.DataFrame(np.concatenate((num_real_data_train_np, cat_real_data_train_oh), axis=1)).astype(float),
        pd.DataFrame(np.concatenate((num_real_data_test_np, cat_real_data_test_oh), axis=1)).astype(float),
        pd.DataFrame(np.concatenate((num_synthetic_data_np, cat_synthetic_data_oh), axis=1)).astype(float),
    )


def distance_to_closest_record_score(
    synthetic_data: pd.DataFrame,
    real_data_train: pd.DataFrame,
    real_data_test: pd.DataFrame,
    norm: NormType = NormType.L1,
    batch_size: int = 1000,
    device: torch.device = DEVICE,
) -> float:
    """
    Compute the distance to closest record (DCR) score for the ``synthetic_data``. DCR is the distance
    between a synthetic datapoint and its nearest real datapoint. DCR equal to zero means that the synthetic
    data is at a higher risk of privacy leakage, while higher DCR values mean less risk of privacy leakage.

    This function computes the DCR of each synthetic datapoint to real data points in two different sets: training and
    test. Then, it returns the proportion of synthetic data points that are closer to the training dataset than the
    test dataset. If the size of the training and test datasets are equal, this score should ideally be indicating
    that the model has not over fit to training data and the synthetic data points are not memorized copies of training
    data. If the size of the training and test datasets are different, the ideal value for this score is #Train /
    (#Train + #Test).

    NOTE: The dataframes provided should already have been pre-processed into numerical values for each column. That
    is, for example, the categorical variables should already have been one-hot encoded and the numerical values
    normalized in some way. This can be done via the ``preprocess_for_distance_to_closest_record_score`` function

    Args:
        synthetic_data: Dataframe containing synthetically generated data for which we want to derive a DCR score.
            This dataframe should already have been preprocessed as in the note above.
        real_data_train: Dataframe containing real data that was used to train the model that generated the provided
            synthetic data. This dataframe should already have been preprocessed as in the note above.
        real_data_test: Dataframe containing real data that was NOT used to train the model that generated the provided
            synthetic data. This dataframe should already have been preprocessed as in the note above.
        norm: What kind of norm to use for the distance calculations.
        batch_size: Batch size used to compute the DCR iteratively. Just needed to manage memory. Defaults to 1000.
        device: What device the tensors should be sent to in order to perform the calculations. Defaults to DEVICE.

    Returns:
        DCR score.
    """
    real_data_train_tensor = torch.tensor(real_data_train.to_numpy()).to(device)
    real_data_test_tensor = torch.tensor(real_data_test.to_numpy()).to(device)
    synthetic_data_tensor = torch.tensor(synthetic_data.to_numpy()).to(device)

    dcr_train = []
    dcr_test = []

    # Assumes that the tensors are 2D and arranged (n_samples, data dimension)
    for start_index in tqdm(range(0, synthetic_data_tensor.size(0), batch_size)):
        end_index = min(start_index + batch_size, synthetic_data_tensor.size(0))
        synthetic_data_batch = synthetic_data_tensor[start_index:end_index]

        # Calculate distances for real and test data in smaller batches
        dcr_train_batch = minimum_distances(synthetic_data_batch, real_data_train_tensor, batch_size, norm)
        dcr_test_batch = minimum_distances(synthetic_data_batch, real_data_test_tensor, batch_size, norm)

        dcr_train.append(dcr_train_batch)
        dcr_test.append(dcr_test_batch)

    dcr_train_torch = torch.cat(dcr_train)
    dcr_test_torch = torch.cat(dcr_test)

    records_closer_to_train = (dcr_train_torch < dcr_test_torch).long().sum()

    score = records_closer_to_train / dcr_train_torch.shape[0]
    log(INFO, f"Distance to Closest Record Score = {score}")
    return score.item()
