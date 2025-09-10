from enum import Enum
from logging import INFO
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from midst_toolkit.common.logger import log
from midst_toolkit.evaluation.generation.quality_metric_base import QualityMetricBase
from midst_toolkit.evaluation.generation.utils import extract_columns_based_on_meta_info


class NormType(Enum):
    L1 = "l1"
    L2 = "l2"


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def compute_l1_distance(
    target_data: torch.Tensor, reference_data: torch.Tensor, skip_diagonal: bool = False
) -> torch.Tensor:
    """
    Compute the smallest l1 distance between each point in the target data tensor compared to all points in the
    reference data tensor.

    Args:
        target_data: Tensor of target data. Assumed to be a 2D tensor with batch size first, followed by
            data dimension.
        reference_data: Tensor of reference data. Assumed to be a 2D tensor with batch size first, followed by
            data dimension.
        skip_diagonal: Whether or not to skip computations on diagonal of distance matrix. This is generally only used
            when ``target_data`` and ``reference_data`` are the same set. In this case, the diagonal elements are the
            distance of the point from itself (which is 0). Defaults to False.

    Returns:
        A 1D tensor containing the l1 minimum distances between each data point in the target data and all points in
        the reference data. Order will be the same as the target data.
    """
    assert target_data.ndim == 2 and reference_data.ndim == 2, "Target and Reference data tensors should be 2D"
    assert target_data.shape[1] == reference_data.shape[1], "Data dimensions do not match for the provided tensors"

    # For target_data (n_target_points, data_dim), and reference_data (n_ref_points, data_dim), this subtracts
    # every point in reference_data from every point in target_data to create a tensor of shape
    # (n_target_points, n_ref_points, data_dim).
    point_differences = target_data[:, None] - reference_data
    distances = (point_differences).abs().sum(dim=2)

    # Minimum distance of points in n_target_points compared to all other points in reference_data.
    if not skip_diagonal:
        min_batch_distances, _ = distances.min(dim=1)
        return min_batch_distances

    # Bottom two distances, because one of them might be the reference point to itself.
    min_batch_distances, _ = torch.topk(distances, 2, dim=1, largest=False)
    return min_batch_distances


def compute_l2_distance(
    target_data: torch.Tensor, reference_data: torch.Tensor, skip_diagonal: bool = False
) -> torch.Tensor:
    """
    Compute the smallest l2 distance between each point in the target data tensor compared to all points in the
    reference data tensor.

    Args:
        target_data: Tensor of synthetic data. Assumed to be a 2D tensor with batch size first, followed by
            data dimension.
        reference_data: Tensor of synthetic data. Assumed to be a 2D tensor with batch size first, followed by
            data dimension.
        skip_diagonal: Whether or not to skip computations on diagonal of distance matrix. This is generally only used
            when ``target_data`` and ``reference_data`` are the same set. In this case, the diagonal elements are the
            distance of the point from itself (which is 0). Defaults to False.

    Returns:
        A 1D tensor containing the l2 minimum distances between each data point in the target data and all points in
        the reference data. Order will be the same as the target data.
    """
    assert target_data.ndim == 2 and reference_data.ndim == 2, "Target and Reference data tensors should be 2D"
    assert target_data.shape[1] == reference_data.shape[1], "Data dimensions do not match for the provided tensors"
    # For target_data (n_target_points, data_dim), and reference_data (n_reference_points, data_dim), this subtracts
    # every point in reference_data from every point in target_data to create a tensor of shape
    # (n_target_points, n_reference_points, data_dim).
    point_differences = target_data[:, None] - reference_data
    distances = torch.sqrt(torch.pow(point_differences, 2.0).sum(dim=2))

    # Minimum distance of points in n_target_points compared to all other points in reference_data.
    if not skip_diagonal:
        min_batch_distances, _ = distances.min(dim=1)
        return min_batch_distances

    # Bottom two distances, because one of them might be the reference point to itself.
    min_batch_distances, _ = torch.topk(distances, 2, dim=1, largest=False)
    return min_batch_distances


def minimum_distances(
    target_data: torch.Tensor,
    reference_data: torch.Tensor,
    batch_size: int | None = None,
    norm: NormType = NormType.L1,
    skip_diagonal: bool = False,
) -> torch.Tensor:
    """
    Function to calculate minimum distances between each point in the target data to those of the reference data
    provided. This can be done in batches if specified. Otherwise, the entire computation is done at once.

    Args:
        target_data: The complete set of target data, stacked as a tensor with shape (n_samples, data dimension).
        reference_data: The complete set of reference data, stacked as a tensor with shape (n_samples, data dimension).
        batch_size: Size of the batches to facilitate computing the minimum distances, if specified. Defaults to None.
        norm: Which type of norm to use as the distance metric. Defaults to NormType.L1.
        skip_diagonal: Whether or not to skip computations on diagonal of distance matrix. This is generally only used
            when ``target_data`` and ``reference_data`` are the same set. In this case, the diagonal elements are the
            distance of the point from itself (which is 0). Defaults to False.

    Returns:
        A 1D tensor with the minimum distances. Should be of length n_samples. Order will be the same as
        ``target_data.``
    """
    if batch_size is None:
        # If batch size isn't specified, do it all at once.
        batch_size = target_data.size(0)

    # Create a minimum distance for each synthetic data sample
    if skip_diagonal:
        min_distances = torch.full((target_data.size(0), 2), float("inf"), device=target_data.device)
    else:
        min_distances = torch.full((target_data.size(0),), float("inf"), device=target_data.device)
    # Iterate through the real data in batches and compute distances
    for start_index in range(0, reference_data.size(0), batch_size):
        end_index = min(start_index + batch_size, reference_data.size(0))
        reference_data_batch = reference_data[start_index:end_index]

        if norm is NormType.L1:
            min_batch_distances = compute_l1_distance(target_data, reference_data_batch, skip_diagonal)
        elif norm is NormType.L2:
            min_batch_distances = compute_l2_distance(target_data, reference_data_batch, skip_diagonal)
        else:
            raise ValueError(f"Unrecognized norm type: {str(norm)}")
        if not skip_diagonal:
            min_distances = torch.minimum(min_distances, min_batch_distances)
        else:
            combined_distances = torch.cat((min_distances, min_batch_distances), dim=1)
            min_distances, _ = torch.topk(combined_distances, 2, dim=1, largest=False)
    if skip_diagonal:
        # Smallest distance should be point to itself. Second smallest is the rest.
        return min_distances[:, 1]
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


class DistanceToClosestRecordScore(QualityMetricBase):
    def __init__(self, norm: NormType = NormType.L1, batch_size: int = 1000, device: torch.device = DEVICE):
        """
        A class to compute the distance to closest record (DCR) score for the ``synthetic_data``. Here, DCR is
        defined as the distance between a synthetic datapoint and its nearest real datapoint. DCR equal to zero means
        that the synthetic data is at a higher risk of privacy leakage, while higher DCR values mean less risk of
        privacy leakage.

        This class computes the DCR of each synthetic datapoint to real data points in two different sets:

        - Training (real) data used to train the model that generated the synthetic data.
        - Holdout (real) data from the same distribution as the training data but that was NOT used to train the model.

        It returns the proportion of synthetic data points that are closer to the training dataset than the
        holdout dataset. If the size of the training and holdout datasets are equal, this score should ideally be
        indicating that the model has not over fit to training data and the synthetic data points are not memorized
        copies of training data. If the size of the training and holdout datasets are different, the ideal value for
        this score is # ``real_data`` / (# ``real_data`` + # ``holdout_data``).

        Args:
            norm: Determines what norm the distances are computed in. Defaults to NormType.L1.
            batch_size: Batch size used to compute the DCR iteratively. Just needed to manage memory. Defaults to 1000.
            device: What device the tensors should be sent to in order to perform the calculations. Defaults to DEVICE.
        """
        self.norm = norm
        self.batch_size = batch_size
        self.device = device

    def compute(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, holdout_data: pd.DataFrame | None = None
    ) -> dict[str, float]:
        """
        Computes the Distance to closest record (DCR) score between the synthetic data and two reference datasets. The
        ``real_data`` dataframe should represent TRAINING data for the model that generated the synthetic data
        while the ``holdout_data`` dataframe should represent heldout data, ideally from the same distribution as
        ``real_data``, that was NOT used to train that model.

        Here, the DCR score is the ratio of synthetic points that are closer to real_data to the combined size of
        ``real_data`` and ``holdout_data``. Ideally, this would be proportionally to randomly selecting a point
        from the combined datasets (# ``real_data`` / (# ``real_data`` + # ``holdout_data``)).

        NOTE: The dataframes provided should already have been pre-processed into numerical values for each column.
        That is, for example, the categorical variables should already have been one-hot encoded and the numerical
        values normalized in some way. This can be done via the ``preprocess_for_distance_to_closest_record_score``
        function

        Args:
            real_data: Real data that was used to train the model that generated the ``synthetic_data``.
            synthetic_data: Synthetic data generated by a model that was trained on ``real_data``.
            holdout_data: Real data that was NOT used to train the generating model. Defaults to None.

        Returns:
            DCR Score
        """
        assert holdout_data is not None, "For DCR score calculations, a holdout dataset is required"

        real_data_train_tensor = torch.tensor(real_data.to_numpy()).to(self.device)
        real_data_test_tensor = torch.tensor(holdout_data.to_numpy()).to(self.device)
        synthetic_data_tensor = torch.tensor(synthetic_data.to_numpy()).to(self.device)

        dcr_train = []
        dcr_test = []

        # Assumes that the tensors are 2D and arranged (n_samples, data dimension)
        for start_index in tqdm(range(0, synthetic_data_tensor.size(0), self.batch_size)):
            end_index = min(start_index + self.batch_size, synthetic_data_tensor.size(0))
            synthetic_data_batch = synthetic_data_tensor[start_index:end_index]

            # Calculate distances for real and test data in smaller batches
            dcr_train_batch = minimum_distances(
                synthetic_data_batch, real_data_train_tensor, self.batch_size, self.norm
            )
            dcr_test_batch = minimum_distances(synthetic_data_batch, real_data_test_tensor, self.batch_size, self.norm)

            dcr_train.append(dcr_train_batch)
            dcr_test.append(dcr_test_batch)

        dcr_train_torch = torch.cat(dcr_train)
        dcr_test_torch = torch.cat(dcr_test)

        records_closer_to_train = (dcr_train_torch < dcr_test_torch).long().sum()

        score = records_closer_to_train / dcr_train_torch.shape[0]
        log(INFO, f"Distance to Closest Record Score = {score}")
        return {"dcr_score": score.item()}


class MedianDistanceToClosestRecordScore(QualityMetricBase):
    def __init__(self, norm: NormType = NormType.L1, batch_size: int = 1000, device: torch.device = DEVICE):
        """
        A metric to compute the Median Distance to Closest Record (Median DCR) metric as described in:
        https://arxiv.org/pdf/2404.15821.

        This calculation uses the same minimum distance to the real training data as in the
        ``distance_to_closest_record_score`` implementation, but then computes the minimum inter-record distances from
        the real training data with itself rather than using a holdout dataset of real values. The ratio of the median
        minimum distance for synthetic to real data to the median minimum distance of real to real is returned.

        Args:
            norm: Determines what norm the distances are computed in. Defaults to NormType.L1.
            batch_size: Batch size used to compute the DCR iteratively. Just needed to manage memory. Defaults to 1000.
            device: What device the tensors should be sent to in order to perform the calculations. Defaults to DEVICE.
        """
        self.norm = norm
        self.batch_size = batch_size
        self.device = device

    def compute(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict[str, float]:
        """
        Implementing the Median Distance to Closest Record (Median DCR) metric as described in:
        https://arxiv.org/pdf/2404.15821.

        This calculation uses the same minimum distance to the real training data as in the
        distance_to_closest_record_score implementation, but then computes the minimum inter-record distances from the
        real training data with itself. The ratio of the median minimum distance for synthetic to real data to the
        median minimum distance of real to real is returned.

        NOTE: The dataframes provided should already have been pre-processed into numerical values for each column.
        That is, for example, the categorical variables should already have been one-hot encoded and the numerical
        values normalized in some way. This can be done via the ``preprocess_for_distance_to_closest_record_score``
        function.

        Args:
            real_data: Dataframe containing real data that was used to train the model that generated the provided
                synthetic data. This dataframe should already have been preprocessed as in the note above.
            synthetic_data: Dataframe containing synthetically generated data for which we want to derive a DCR score.
                This dataframe should already have been preprocessed as in the note above.

        Returns:
            Median DCR score
        """
        real_data_tensor = torch.tensor(real_data.to_numpy()).to(self.device)
        synthetic_data_tensor = torch.tensor(synthetic_data.to_numpy()).to(self.device)

        dcr_synthetic_to_real = []
        dcr_real_to_real = []

        # Assumes that the tensors are 2D and arranged (n_samples, data dimension)
        for start_index in tqdm(range(0, synthetic_data_tensor.size(0), self.batch_size)):
            end_index = min(start_index + self.batch_size, synthetic_data_tensor.size(0))
            synthetic_data_batch = synthetic_data_tensor[start_index:end_index]

            # Calculate distances for synthetic data to real data in smaller batches
            dcr_synthetic_to_real_batch = minimum_distances(
                synthetic_data_batch, real_data_tensor, self.batch_size, self.norm, skip_diagonal=False
            )
            dcr_synthetic_to_real.append(dcr_synthetic_to_real_batch)

            # Assumes that the tensors are 2D and arranged (n_samples, data dimension)
        for start_index in tqdm(range(0, real_data_tensor.size(0), self.batch_size)):
            end_index = min(start_index + self.batch_size, real_data_tensor.size(0))
            real_data_batch = real_data_tensor[start_index:end_index]

            # Calculate distances for synthetic data to real data in smaller batches
            dcr_real_to_real_batch = minimum_distances(
                real_data_batch, real_data_tensor, self.batch_size, self.norm, skip_diagonal=True
            )
            dcr_real_to_real.append(dcr_real_to_real_batch)

        dcr_synthetic_to_real_torch = torch.cat(dcr_synthetic_to_real)
        dcr_real_to_real_torch = torch.cat(dcr_real_to_real)

        median_dcr_score = (
            torch.median(dcr_synthetic_to_real_torch).item() / torch.median(dcr_real_to_real_torch).item()
        )
        return {"median_dcr_score": median_dcr_score}
