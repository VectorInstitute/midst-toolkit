import math
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from midst_toolkit.common.variables import DEVICE
from midst_toolkit.evaluation.metrics_base import MetricBase
from midst_toolkit.evaluation.privacy.distance_preprocess import preprocess_for_distance_computation
from midst_toolkit.evaluation.privacy.distance_utils import NormType


class NearestNeighborDistanceRatio(MetricBase):
    def __init__(
        self,
        norm: NormType = NormType.L2,
        batch_size: int = 1000,
        reference_batch_size: int = 5000,  # NEW: separate batch size for reference data
        device: torch.device = DEVICE,
        meta_info: dict[str, Any] | None = None,
        do_preprocess: bool = False,
        epsilon: float = 1e-8,
        use_cpu: bool = False,  # NEW: option to force CPU
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
        be done within ``compute`` if ``do_preprocess`` is True and ``meta_info`` has been provided.

        Args:
            norm: Determines what norm the distances are computed in. Defaults to NormType.L2.
            batch_size: Batch size used to compute the NNDR iteratively for target data. Just needed to manage memory. 
                Defaults to 1000.
            reference_batch_size: Batch size for processing reference data. Defaults to 5000.
            device: What device the tensors should be sent to in order to perform the calculations. Defaults to
                "cuda" if CUDA is available, "cpu" otherwise.
            meta_info: This is only required/used if ``do_preprocess`` is True. JSON with meta information about the
                columns and their corresponding types that should be considered. At minimum, it should have the keys
                'num_col_idx' and 'cat_col_idx'. If 'target_col_idx' is specified then 'task_type' must also exist.
                If None, then no preprocessing is expected to be done. Defaults to None.
            do_preprocess: Whether or not to preprocess the dataframes before performing the NNDR calculations.
                Preprocessing is performed with the ``preprocess`` function of ``distance_preprocess.py``. Note,
                ``meta_info`` must be provided in order  to perform the appropriate preprocessing steps. Defaults to
                False.
            epsilon: Regularization term that ensures that we do not divide by 0. Defaults to 1e-8
            use_cpu: If True, forces computation on CPU regardless of GPU availability. Defaults to False.
        """
        self.norm = norm
        self.batch_size = batch_size
        self.reference_batch_size = reference_batch_size
        # Force CPU if requested
        if use_cpu:
            self.device = torch.device("cpu")
            print("INFO: Forcing CPU mode for NNDR computation")
        else:
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
        positive, the more the synthetic data may reveal about the original training set.

        NOTE: The dataframes provided need to be pre-processed into numerical values for each column in some way. That
        is, for example, the categorical variables may be one-hot encoded and the numerical values normalized in
        some way. This can be done via the ``preprocess`` function in ``distance_preprocess.py`` beforehand or it can
        be done within ``compute`` if ``do_preprocess`` is True and ``meta_info`` has been provided.

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
                real_data, synthetic_data = preprocess_for_distance_computation(
                    self.meta_info, real_data, synthetic_data
                )
            else:
                real_data, synthetic_data, holdout_data = preprocess_for_distance_computation(
                    self.meta_info, real_data, synthetic_data, holdout_data
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

    def _compute_l2_distances_batched(
        self, target_data: torch.Tensor, reference_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute L2 distances between target and reference data in batches to avoid OOM errors.
        
        Args:
            target_data: Tensor of shape (n_target, n_features)
            reference_data: Tensor of shape (n_reference, n_features)
            
        Returns:
            Tensor of shape (n_target, n_reference) containing pairwise L2 distances
        """
        n_target = target_data.size(0)
        n_reference = reference_data.size(0)
        
        # Initialize distance matrix on CPU first to save GPU memory
        distances = torch.zeros((n_target, n_reference), dtype=target_data.dtype, device='cpu')
        
        # Process reference data in batches
        for ref_start in range(0, n_reference, self.reference_batch_size):
            ref_end = min(ref_start + self.reference_batch_size, n_reference)
            reference_batch = reference_data[ref_start:ref_end]
            
            # Compute squared differences: (n_target, 1, n_features) - (1, n_ref_batch, n_features)
            # This broadcasts to (n_target, n_ref_batch, n_features)
            squared_diff = (target_data.unsqueeze(1) - reference_batch.unsqueeze(0)) ** 2
            
            # Sum over features and take sqrt: (n_target, n_ref_batch)
            batch_distances = torch.sqrt(torch.sum(squared_diff, dim=2))
            
            # Move to CPU to save GPU memory
            distances[:, ref_start:ref_end] = batch_distances.cpu()
            
            # Clear GPU cache
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Move back to device for further processing
        return distances.to(self.device)

    def _compute_top_k_distances_batched(
        self, target_data: torch.Tensor, reference_data: torch.Tensor, top_k: int = 2
    ) -> torch.Tensor:
        """
        Compute top-k smallest distances between target and reference data in a memory-efficient way.
        
        Args:
            target_data: Tensor of shape (n_target, n_features)
            reference_data: Tensor of shape (n_reference, n_features)
            top_k: Number of smallest distances to return
            
        Returns:
            Tensor of shape (n_target, top_k) containing the k smallest distances for each target point
        """
        if self.norm == NormType.L2:
            distances = self._compute_l2_distances_batched(target_data, reference_data)
        else:
            raise NotImplementedError(f"Norm type {self.norm} not implemented for batched computation")
        
        # Get top-k smallest distances
        top_k_distances, _ = torch.topk(distances, k=top_k, dim=1, largest=False, sorted=True)
        
        return top_k_distances

    def _compute_mean_nearest_neighbor_distance_ratio(
        self, target_tensor: torch.Tensor, reference_tensor: torch.Tensor
    ) -> tuple[float, float]:
        ratios = []
        # Assumes that the tensors are 2D and arranged (n_samples, data dimension)
        print(f"Computing NNDR for {target_tensor.size(0)} target samples against {reference_tensor.size(0)} reference samples")
        print(f"Using batch_size={self.batch_size} for target, reference_batch_size={self.reference_batch_size}")
        
        for start_index in tqdm(range(0, target_tensor.size(0), self.batch_size), desc="Processing target batches"):
            end_index = min(start_index + self.batch_size, target_tensor.size(0))
            target_data_batch = target_tensor[start_index:end_index]

            # Calculate top-2 distances using batched computation
            top_2_distances = self._compute_top_k_distances_batched(target_data_batch, reference_tensor, top_k=2)
            ratios.append(top_2_distances[:, 0] / (top_2_distances[:, 1] + self.epsilon))

        all_ratios = torch.cat(ratios)
        mean_ratios = float(torch.mean(all_ratios).item())
        ratios_standard_error = torch.std(all_ratios).item() / math.sqrt(len(all_ratios))

        return mean_ratios, ratios_standard_error