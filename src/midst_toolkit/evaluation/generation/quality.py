import logging
from typing import Any

import pandas as pd
from synthcity.metrics import eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def synthcity_alpha_precision_metrics(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame, naive_only: bool = True
) -> dict[str, Any]:
    """
    Computes a number of quality metrics comparing the synthetic data to ground truth data using the Synthcity library
    This function uses the AlphaPrecision class in that library, which computes the alpha-precision, beta-recall, and
    authenticity scores between the two datasets. If the ``naive_only`` boolean is True, then only the "naive" metrics
    are reported.

    Args:
        real_data: Real data that the synthetic data is meant to mimic/replace.
        synthetic_data: Synthetic data to be compared against the provided real data.
        naive_only: If True, then only the "naive" metrics are reported. Defaults to True.

    Returns:
        A dictionary containing the computed scores using the AlphaPrecision class in the Synthcity library
    """
    # Wrap the dataframes in a Synthcity compatible dataloader
    real_data_loader = GenericDataLoader(real_data)
    synthetic_data_loader = GenericDataLoader(synthetic_data)

    quality_evaluator = eval_statistical.AlphaPrecision()
    quality_results = quality_evaluator.evaluate(real_data_loader, synthetic_data_loader)
    if naive_only:
        quality_results = {
            # Filter to results associated with naive checks
            key: metric
            for (key, metric) in quality_results.items()
            if "naive" in key
        }

    LOGGER.info(
        f"Naive Alpha Precision: {quality_results['delta_precision_alpha_naive']}\n"
        f"Naive Beta Recall: {quality_results['delta_coverage_beta_naive']}"
    )

    return quality_results
