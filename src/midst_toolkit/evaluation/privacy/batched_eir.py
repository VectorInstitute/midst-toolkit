from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import entropy
from syntheval.metrics.core.metric import MetricClass
from syntheval.utils.nn_distance import _knn_distance
from tqdm.auto import tqdm


def _column_entropy(labels: list | np.ndarray) -> np.number:
    """
    Compute the entropy of a single column of labels.

    Args:
        labels: One-dimensional collection of labels. Values are rounded
            before computing entropy.

    Returns:
        The entropy of the distribution of rounded labels.
    """
    _, counts = np.unique(np.round(labels), return_counts=True)
    return entropy(counts)


def batched_reference_knn(
    query_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    categorical_columns: list[int],
    nn_distance_metric: Literal["gower", "euclid"],
    weights: np.ndarray,
    ref_batch_size: int = 128,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute nearest neighbor distances from the points in query_df to reference_df in a memory-efficient way.

    Instead of comparing all query rows to all reference rows at once, the reference DataFrame
    is split into batches. For each batch:
      1. Compute the distances from all query rows to the current reference_df batch.
      2. Keep track of the smallest distance per query row across all batches.

    Args:
        query_df : The data points for which nearest neighbor distances are computed.
        reference_df : The data points used as the reference for computing distances.
        categorical_columns : Indices of categorical columns.
        nn_distance_metric : Distance metric to use for nearest neighbor distance computation. Possible values are the
                             Gower distance metric ('gower') and the Euclidean distance metric ('euclid').
        weights : Feature weights to apply when computing distances.
        ref_batch_size :  Number of reference rows per batch.
        show_progress : Whether to display a progress bar over reference batches.

    Returns:
        Array of nearest neighbor distance per query row after considering all reference batches.
    """
    query_df_size = len(query_df)

    # Initizalizing a list of best distances with np.inf so they can be replaced with the actual best distances later.
    nearest_neighbor_distance = np.full(query_df_size, np.inf, dtype=float)

    iterator: Iterable[int]
    if show_progress:
        iterator = tqdm(
            range(0, len(reference_df), ref_batch_size),
            total=(len(reference_df) + ref_batch_size - 1) // ref_batch_size,
            desc="Computing nearest neighbor distances from real/holdout dataset to synthetic dataset.",
        )
    else:
        iterator = range(0, len(reference_df), ref_batch_size)

    for start in iterator:
        end = min(start + ref_batch_size, len(reference_df))
        ref_batch = reference_df.iloc[start:end]

        # compute distances for each row of the reference batch to its closest neigbour in ref_batch
        # hardcoding of k=1 refers to only needing to compute the distance to the closest neighbor.
        batch_distances = _knn_distance(query_df, ref_batch, categorical_columns, 1, nn_distance_metric, weights)[0]

        # keep smallest per query row
        nearest_neighbor_distance = np.minimum(nearest_neighbor_distance, batch_distances)

    return nearest_neighbor_distance


class EpsilonIdentifiability(MetricClass):  # type: ignore[misc]
    def name(self) -> str:
        """
        Returns the identifier of the metric.

        Returns:
            "eps_risk"
        """
        return "eps_risk"

    def type(self) -> str:
        """
        Returns the type of the evaluation metric.

        Returns:
            "privacy"
        """
        return "privacy"

    def evaluate(self) -> dict[str, float]:
        """
        Compute epsilon-identifiability risk and privacy loss.

        The epsilon-identifiability risk (eps_risk) is defined as the fraction of real
        records whose nearest neighbor in the synthetic dataset is closer than their
        nearest neighbor in the real dataset, using an entropy-weighted distance metric.

        If holdout data is provided, the privacy loss (priv_loss) is computed as the
        difference between the identifiability risk on the training data and the
        identifiability risk on the holdout data.

        Returns:
            dict:
                - 'eps_risk': Fraction of real records vulnerable to re-identification.
                - 'priv_loss': Difference between training and holdout identifiability risks
                (only present if holdout data is not None).
        """
        np_real_data = np.asarray(self.real_data)
        real_size, n_feautures = np_real_data.shape

        # Column entropies → weights (inverted)
        weights = [_column_entropy(np_real_data[:, feauture]) for feauture in range(n_feautures)]
        weights_adjusted = 1 / (np.array(weights) + 1e-16)

        # internal (original syntheval logic)
        # hardcoding of k=1 refers to only needing to compute the distance to the closest neighbor.
        internal_distances = _knn_distance(
            self.real_data,
            self.real_data,
            self.cat_cols,
            1,
            self.nn_dist,
            weights_adjusted,
        )[0]

        # external (batched)
        external_distances = batched_reference_knn(
            self.real_data,
            self.synt_data,
            self.cat_cols,
            self.nn_dist,
            weights_adjusted,
        )

        real_data_distance_differences = external_distances - internal_distances
        identifiability_risk = np.sum(real_data_distance_differences < 0) / float(real_size)
        self.results["eps_risk"] = identifiability_risk

        if self.hout_data is not None:
            # internal (original syntheval logic)
            # hardcoding of k=1 refers to only needing to compute the distance to the closest neighbor.
            hout_internal_distances = _knn_distance(
                self.hout_data, self.hout_data, self.cat_cols, 1, self.nn_dist, weights_adjusted
            )[0]

            # external (batched)
            hout_external_distances = batched_reference_knn(
                self.hout_data,
                self.synt_data,
                self.cat_cols,
                self.nn_dist,
                weights_adjusted,
            )

            holdout_data_distance_differences = hout_external_distances - hout_internal_distances
            hout_identifiability_risk = np.sum(holdout_data_distance_differences < 0) / float(len(self.hout_data))

            self.results["priv_loss"] = self.results["eps_risk"] - hout_identifiability_risk

        return self.results

    def format_output(self) -> str:
        """Format the output for printing."""
        string = f"| Epsilon identifiability risk             :   {self.results['eps_risk']:.4f}           |"
        if self.results != {} and self.hout_data is not None:
            string += f"\n| Privacy loss (diff. in eps. risk)        :   {self.results['priv_loss']:.4f}           |"
        return string

    def normalize_output(self) -> list[dict[str, Any]] | None:
        """
        Convert computed privacy metrics into a standardized list of dictionaries.

        Each dictionary contains:
            - 'metric': The metric identifier
            - 'val': The raw metric value

        The metrics included are:
            - 'eps_identif_risk': The epsilon-identifiability risk of the real data
            - 'priv_loss_eps': The difference in epsilon risk between training and holdout
            data (only included if holdout data is provided)

        If the evaluation has not been run yet (i.e., results are empty),
        the method returns None.

        Returns:
            A list of metric dictionaries if results are available;
            otherwise, None.
        """
        if self.results == {}:
            return None

        output = [
            {
                "metric": "eps_identif_risk",
                "val": self.results["eps_risk"],
            }
        ]

        if self.hout_data is not None:
            output.append(
                {
                    "metric": "priv_loss_eps",
                    "val": self.results["priv_loss"],
                }
            )

        return output
