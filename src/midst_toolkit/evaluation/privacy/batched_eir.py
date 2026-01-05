from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import entropy
from syntheval.metrics.core.metric import MetricClass
from syntheval.utils.nn_distance import _knn_distance
from tqdm.auto import tqdm


def _column_entropy(labels: list | np.ndarray) -> np.number:
    """Compute the entropy of a single column."""
    value, counts = np.unique(np.round(labels), return_counts=True)
    return entropy(counts)


def batched_reference_knn(
    query_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    cat_cols: list[int],
    nn_dist: str,
    weights: np.ndarray,
    ref_batch_size: int = 128,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute k-nearest neighbor distances from query rows to reference rows in a memory-efficient way.

    Instead of comparing all query rows to all reference rows at once, the reference DataFrame
    is split into batches. For each batch:
      1. Compute the distances from all query rows to the current batch.
      2. Keep track of the smallest distance per query row across all batches.

    Args:
        query_df : The data points for which kNN distances are computed.
        reference_df : The data points used as the reference for computing distances.
        cat_cols : Indices of categorical columns.
        nn_dist : Distance metric to use for nearest neighbor computation.
        weights : Feature weights to apply when computing distances.
        ref_batch_size :  Number of reference rows per batch.
        show_progress : Whether to display a progress bar over reference batches.

    Returns :
        Array of minimum distances per query row after considering all reference batches.
    """
    n_query = len(query_df)

    # best distances so far = +inf
    best_d = np.full(n_query, np.inf, dtype=float)

    iterator: Iterable[int]
    if show_progress:
        iterator = tqdm(
            range(0, len(reference_df), ref_batch_size),
            total=(len(reference_df) + ref_batch_size - 1) // ref_batch_size,
            desc="Computing ref-batched kNN distances",
        )
    else:
        iterator = range(0, len(reference_df), ref_batch_size)

    for start in iterator:
        end = min(start + ref_batch_size, len(reference_df))
        ref_batch = reference_df.iloc[start:end]

        # compute distances to this reference batch (k=1 → index 0)
        d_batch = _knn_distance(query_df, ref_batch, cat_cols, 1, nn_dist, weights)[0]

        # keep smallest per query row
        best_d = np.minimum(best_d, d_batch)

    return best_d


class EpsilonIdentifiability(MetricClass):  # type: ignore[misc]
    def name(self) -> str:
        """Return the name of the metric."""
        return "eps_risk"

    def type(self) -> str:
        """Return the type of the metric."""
        return "privacy"

    def evaluate(self) -> dict:
        """Compute the Epsilon Identifiability Risk and Privacy Loss."""
        real = np.asarray(self.real_data)
        no, x_dim = real.shape

        # Column entropies → weights (inverted)
        weights = [_column_entropy(real[:, i]) for i in range(x_dim)]
        weights_adjusted = 1 / (np.array(weights) + 1e-16)

        # INTERNAL KNN: REAL → REAL
        in_dists = _knn_distance(
            self.real_data,
            self.real_data,
            self.cat_cols,
            1,
            self.nn_dist,
            weights_adjusted,
        )[0]

        # EXTERNAL KNN: REAL → SYNTHETIC (safe to batch reference)
        ext_dists = batched_reference_knn(
            self.real_data,
            self.synt_data,
            self.cat_cols,
            self.nn_dist,
            weights_adjusted,
        )

        r_diff = ext_dists - in_dists
        identifiability = np.sum(r_diff < 0) / float(no)
        self.results["eps_risk"] = identifiability

        if self.hout_data is not None:
            # INTERNAL: HOUT → HOUT (original logic)
            hout_in = _knn_distance(self.hout_data, self.hout_data, self.cat_cols, 1, self.nn_dist, weights_adjusted)[
                0
            ]

            # EXTERNAL: HOUT → SYNTHETIC (batched)
            hout_ext = batched_reference_knn(
                self.hout_data,
                self.synt_data,
                self.cat_cols,
                self.nn_dist,
                weights_adjusted,
            )

            hout_diff = hout_ext - hout_in
            hout_val = np.sum(hout_diff < 0) / float(len(self.hout_data))

            self.results["priv_loss"] = self.results["eps_risk"] - hout_val

        return self.results

    def format_output(self) -> str:
        """Format the output for printing."""
        string = f"| Epsilon identifiability risk             :   {self.results['eps_risk']:.4f}           |"
        if self.results != {} and self.hout_data is not None:
            string += f"\n| Privacy loss (diff. in eps. risk)        :   {self.results['priv_loss']:.4f}           |"
        return string

    def normalize_output(self) -> list | None:
        """Standardize the output format."""
        if self.results == {}:
            return None

        output = [
            {
                "metric": "eps_identif_risk",
                "dim": "p",
                "val": self.results["eps_risk"],
                "n_val": 1 - self.results["eps_risk"],
            }
        ]

        if self.hout_data is not None:
            output.append(
                {
                    "metric": "priv_loss_eps",
                    "dim": "p",
                    "val": self.results["priv_loss"],
                    "n_val": 1 - abs(self.results["priv_loss"]),
                }
            )

        return output
