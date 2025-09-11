from logging import INFO
from typing import Any, overload

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from midst_toolkit.common.logger import log
from midst_toolkit.evaluation.generation.distance_utils import NormType, minimum_distances
from midst_toolkit.evaluation.generation.quality_metric_base import QualityMetricBase
from midst_toolkit.evaluation.generation.utils import extract_columns_based_on_meta_info


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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
    This function performs preprocessing on Pandas dataframes to prepare for computation of the distance to closest
    record score. Specifically, this function filters the provided raw dataframes to the appropriate numerical and
    categorical columns based on the information of the ``meta_info`` JSON. For the numerical columns, it normalizes
    values by the distance between the largest and smallest value of each column of the ``real_data_train`` numerical
    values. The categorical columns are processed into one-hot encoding columns, where the transformation is fitted
    on the concatenation of columns from each dataset.

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
    return (
        processed_synthetic_data,
        processed_real_data_train,
        pd.DataFrame(np.concatenate((num_real_data_test_np, cat_real_data_test_oh), axis=1)).astype(float),
    )


class DistanceToClosestRecordScore(QualityMetricBase):
    def __init__(
        self,
        norm: NormType = NormType.L1,
        batch_size: int = 1000,
        device: torch.device = DEVICE,
        meta_info: dict[str, Any] | None = None,
        do_preprocess: bool = False,
    ):
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
            meta_info: This is only required/used if ``do_preprocess`` is True. JSON with meta information about the
                columns and their corresponding types that should be considered. At minimum, it should have the keys
                'num_col_idx', 'cat_col_idx', 'target_col_idx', and 'task_type'. If None, then no preprocessing is
                expected to be done. Defaults to None.
            do_preprocess: Whether or not to preprocess the dataframes before performing the DCR computations.
                Preprocessing is performed with the ``preprocess_for_distance_to_closest_record_score`` function
                Defaults to False.
        """
        self.norm = norm
        self.batch_size = batch_size
        self.device = device
        self.do_preprocess = do_preprocess
        if self.do_preprocess and meta_info is None:
            raise ValueError("Preprocessing requires meta_info to be defined, but it is None.")
        self.meta_info = meta_info if meta_info is not None else {}

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

        NOTE: The dataframes provided need to be pre-processed into numerical values for each column in some way. That
        is, for example, the categorical variables should be one-hot encoded and the numerical values normalized in
        some way. This can be done via the ``preprocess_for_distance_to_closest_record_score`` function beforehand or
        it can be done within compute if ``do_preprocess`` is True and ``meta_info`` has been provided.

        Args:
            real_data: Real data that was used to train the model that generated the ``synthetic_data``.
            synthetic_data: Synthetic data generated by a model that was trained on ``real_data``.
            holdout_data: Real data that was NOT used to train the generating model. Defaults to None.

        Returns:
            A dictionary containing the Distance to Closest Record Score in the ``dcr_score`` key. Example:
            { "dcr_score": 0.79 }
        """
        assert holdout_data is not None, "For DCR score calculations, a holdout dataset is required"

        if self.do_preprocess:
            synthetic_data, real_data, holdout_data = preprocess(
                self.meta_info, synthetic_data, real_data, holdout_data
            )

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
    def __init__(
        self,
        norm: NormType = NormType.L1,
        batch_size: int = 1000,
        device: torch.device = DEVICE,
        meta_info: dict[str, Any] | None = None,
        do_preprocess: bool = False,
    ):
        """
        A metric to compute the Median Distance to Closest Record (Median DCR) metric as described in:
        https://arxiv.org/pdf/2404.15821.

        First, the minimum distance from points in the synthetic data to the real data are computed. Next the minimum
        inter-record distances from the real training data with itself are calculated. The ratio of the median minimum
        distance for synthetic to real data to the median minimum distance of real to real is returned.

        Args:
            norm: Determines what norm the distances are computed in. Defaults to NormType.L1.
            batch_size: Batch size used to compute the DCR iteratively. Just needed to manage memory. Defaults to 1000.
            device: What device the tensors should be sent to in order to perform the calculations. Defaults to DEVICE.
            meta_info: This is only required/used if ``do_preprocess`` is True. JSON with meta information about the
                columns and their corresponding types that should be considered. At minimum, it should have the keys
                'num_col_idx', 'cat_col_idx', 'target_col_idx', and 'task_type'. If None, then no preprocessing is
                expected to be done. Defaults to None.
            do_preprocess: Whether or not to preprocess the dataframes before performing the DCR computations.
                Preprocessing is performed with the ``preprocess_for_distance_to_closest_record_score`` function
                Defaults to False.
        """
        self.norm = norm
        self.batch_size = batch_size
        self.device = device
        self.do_preprocess = do_preprocess
        if self.do_preprocess and meta_info is None:
            raise ValueError("Preprocessing requires meta_info to be defined, but it is None.")
        self.meta_info = meta_info if meta_info is not None else {}

    def compute(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict[str, float]:
        """
        Implementing the Median Distance to Closest Record (Median DCR) metric as described in:
        https://arxiv.org/pdf/2404.15821.

        First, the minimum distance from points in the synthetic data to the real data are computed. Next the minimum
        inter-record distances from the real training data with itself are calculated. The ratio of the median minimum
        distance for synthetic to real data to the median minimum distance of real to real is returned.

        NOTE: The dataframes provided need to be pre-processed into numerical values for each column in some way. That
        is, for example, the categorical variables should be one-hot encoded and the numerical values normalized in
        some way. This can be done via the ``preprocess_for_distance_to_closest_record_score`` function beforehand or
        it can be done within compute if ``do_preprocess`` is True and ``meta_info`` has been provided.

        Args:
            real_data: Dataframe containing real data that was used to train the model that generated the provided
                synthetic data. This dataframe should already have been preprocessed as in the note above.
            synthetic_data: Dataframe containing synthetically generated data for which we want to derive a DCR score.
                This dataframe should already have been preprocessed as in the note above.

        Returns:
            A dictionary containing the Median Distance to Closest Record Score in the ``median_dcr_score`` key.
            Example: { "median_dcr_score": 0.79 }
        """
        if self.do_preprocess:
            synthetic_data, real_data = preprocess(self.meta_info, synthetic_data, real_data)

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
