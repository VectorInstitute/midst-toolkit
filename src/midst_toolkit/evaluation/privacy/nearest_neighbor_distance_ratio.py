import math
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from midst_toolkit.evaluation.metrics_base import MetricBase
from midst_toolkit.evaluation.privacy.distance_preprocess import preprocess
from midst_toolkit.evaluation.privacy.distance_utils import NormType, compute_top_k_distances


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class NearestNeighborDistanceRatio(MetricBase):
    def __init__(
        self,
        norm: NormType = NormType.L2,
        batch_size: int = 1000,
        device: torch.device = DEVICE,
        meta_info: dict[str, Any] | None = None,
        do_preprocess: bool = False,
        epsilon: float = 1e-8,
    ):
        """
        This class computes the nearest neighbor distance ratio (NNDR) between synthetic and real datasets. The
        primary, real dataset typically corresponds to the data used to train the model that generated the
        corresponding synthetic dataset. For each point in the synthetic dataset, the top two nearest points in the
        real dataset are computed. The ratio of the two distances (closes/second closest) is computed for all synthetic
        points and averaged for the final score.

        See: https://arxiv.org/pdf/2501.03941

        Intuitively, this measures whether the synthetic points are in "dense" areas of the real data or "sparse"
        regions, potentially endangering outliers. If the area is dense, the two distances will be similar and the
        ratio close to 1. If not, the second closest point may be much farther away, producing a ratio closer to 0.

        If a holdout dataset, composed of real data points that were NOT used to train the generating model, is
        provided the same computation comparing the synthetic data to the holdout set is performed. The difference
        between the two ratios (train and holdout comparisons) is a score comparing the "privacy loss." The more
        positive, the more the synthetic data may reveal about the original training set.

        NOTE: The dataframes provided need to be pre-processed into numerical values for each column in some way. That
        is, for example, the categorical variables may be one-hot encoded and the numerical values normalized in
        some way. This can be done via the ``preprocess`` function in ``distance_preprocess.py`` beforehand or it can
        be done within compute if ``do_preprocess`` is True and ``meta_info`` has been provided.

        Args:
            norm: Determines what norm the distances are computed in. Defaults to NormType.L2.
            batch_size: Batch size used to compute the NNDR iteratively. Just needed to manage memory. Defaults to
                1000.
            device: What device the tensors should be sent to in order to perform the calculations. Defaults to DEVICE.
            meta_info: This is only required/used if ``do_preprocess`` is True. JSON with meta information about the
                columns and their corresponding types that should be considered. At minimum, it should have the keys
                'num_col_idx' and 'cat_col_idx'. If 'target_col_idx' is specified then 'task_type' must also exist.
                If None, then no preprocessing is expected to be done. Defaults to None.
            do_preprocess: Whether or not to preprocess the dataframes before performing the NNDR calculations.
                Preprocessing is performed with the ``preprocess`` function of ``distance_preprocess.py``. Defaults to
                False.
            epsilon: Regularization term that ensures that we do not divide by 0. Defaults to 1e-8
        """
        self.norm = norm
        self.batch_size = batch_size
        self.device = device
        self.do_preprocess = do_preprocess
        if self.do_preprocess and meta_info is None:
            raise ValueError("Preprocessing requires meta_info to be defined, but it is None.")
        self.meta_info = meta_info if meta_info is not None else {}
        self.epsilon = epsilon

    def compute(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, holdout_data: pd.DataFrame | None = None
    ) -> dict[str, float]:
        """
        Computes the nearest neighbor distance ratio (NNDR) between synthetic and real datasets. The primary, real
        dataset typically corresponds to the data used to train the model that generated the corresponding synthetic
        dataset. For each point in the synthetic dataset, the top two nearest points in the real dataset are computed.
        The ratio of the two distances (closes/second closest) is computed for all synthetic points and averaged for
        the final score.

        If a holdout dataset, composed of real data points that were NOT used to train the generating model, is
        provided the same computation comparing the synthetic data to the holdout set is performed. The difference
        between the two ratios (train and holdout comparisons) is a score comparing the "privacy loss." The more
        positive, the more th synthetic data may reveal about the original training set.

        NOTE: The dataframes provided need to be pre-processed into numerical values for each column in some way. That
        is, for example, the categorical variables may be one-hot encoded and the numerical values normalized in
        some way. This can be done via the ``preprocess`` function in ``distance_preprocess.py`` beforehand or it can
        be done within compute if ``do_preprocess`` is True and ``meta_info`` has been provided.

        Args:
            real_data: Real data to which the synthetic data may be compared. In many cases this will be data used
                to TRAIN the model that generated the synthetic data, but not always.
            synthetic_data: Synthetically generated data whose quality is to be assessed.
            holdout_data: Real data to which the synthetic data may also be compared. In many cases this will be data
                was NOT used in training the generating model. If none, then the metrics 'privacy_loss' and
                'privacy_loss_standard_error' are not reported. Defaults to None.

        Returns:
            A dictionary of NNDR results. Regardless of input, the mean of the NNDR values for each synthetic data
            point and standard error of the mean are reported, keyed by 'mean_nndr' and 'nndr_standard_error',
            respectively. If ``holdout_data`` is provided. The difference of the mean nndr using ``real_data`` and
            ``holdout_data`` is reported as 'privacy_loss', along with the pooled standard errors for both
            mean nndr values (key: 'privacy_loss_standard_error').
        """
        if self.do_preprocess:
            if holdout_data is None:
                synthetic_data, real_data = preprocess(self.meta_info, synthetic_data, real_data)
            else:
                synthetic_data, real_data, holdout_data = preprocess(
                    self.meta_info, synthetic_data, real_data, holdout_data
                )

        synthetic_data_tensor = torch.tensor(synthetic_data.to_numpy()).to(self.device)
        real_data_tensor = torch.tensor(real_data.to_numpy()).to(self.device)
        mean_nndr, nndr_standard_error = self._compute_mean_nearest_neighbor_distance_ratio(
            synthetic_data_tensor, real_data_tensor
        )

        result = {
            "mean_nndr": mean_nndr,
            "nndr_standard_error": nndr_standard_error,
        }

        if holdout_data is not None:
            holdout_data_tensor = torch.tensor(holdout_data.to_numpy()).to(self.device)
            mean_nndr_holdout, nndr_standard_error_holdout = self._compute_mean_nearest_neighbor_distance_ratio(
                synthetic_data_tensor, holdout_data_tensor
            )
            result["privacy_loss"] = mean_nndr - mean_nndr_holdout
            result["privacy_loss_standard_error"] = math.sqrt(nndr_standard_error**2 + nndr_standard_error_holdout**2)

        return result

    def _compute_mean_nearest_neighbor_distance_ratio(
        self, target_tensor: torch.Tensor, reference_tensor: torch.Tensor
    ) -> tuple[float, float]:
        ratios = []
        # Assumes that the tensors are 2D and arranged (n_samples, data dimension)
        for start_index in tqdm(range(0, target_tensor.size(0), self.batch_size)):
            end_index = min(start_index + self.batch_size, target_tensor.size(0))
            target_data_batch = target_tensor[start_index:end_index]

            # Calculate top-2 distances for real and test data in smaller batches
            top_2_distances = compute_top_k_distances(target_data_batch, reference_tensor, self.norm, top_k=2)
            ratios.append(top_2_distances[:, 0] / (top_2_distances[:, 1] + self.epsilon))

        all_ratios = torch.cat(ratios)
        mean_ratios = float(torch.mean(all_ratios).item())
        ratios_standard_error = torch.std(all_ratios).item() / math.sqrt(len(all_ratios))

        return mean_ratios, ratios_standard_error
