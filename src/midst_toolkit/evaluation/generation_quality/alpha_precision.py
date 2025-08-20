from logging import INFO
from typing import Any

import pandas as pd
from synthcity.metrics import eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader

from midst_toolkit.common.logger import log


NAIVE_METRIC_SUFFIX = "naive"


def synthcity_alpha_precision_metrics(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame, naive_only: bool = True
) -> dict[str, Any]:
    """
    Computes a number of quality metrics comparing the synthetic data to ground truth data using the Synthcity library.
    This function uses the AlphaPrecision class in that library, which computes the alpha-precision, beta-recall, and
    authenticity scores between the two datasets. If the ``naive_only`` boolean is True, then only the "naive" metrics
    are reported, i.e. metrics with "naive" in their name.

    Args:
        real_data: Real data that the synthetic data is meant to mimic/replace.
        synthetic_data: Synthetic data to be compared against the provided real data.
        naive_only: If True, then only the "naive" metrics are reported. Defaults to True.

    Returns:
        A dictionary containing the computed scores using the AlphaPrecision class in the Synthcity library.
    """
    # Wrap the dataframes in a Synthcity compatible dataloader
    real_data_loader = GenericDataLoader(real_data)
    synthetic_data_loader = GenericDataLoader(synthetic_data)

    quality_evaluator = eval_statistical.AlphaPrecision()
    quality_results = quality_evaluator.evaluate(real_data_loader, synthetic_data_loader)

    # Log results and filter to naive keys if requested
    for metric_key, metric_value in quality_results.items():
        log(INFO, f"{metric_key}: {metric_value}")
        if naive_only and (NAIVE_METRIC_SUFFIX not in metric_key):
            del quality_results[metric_key]

    return quality_results
