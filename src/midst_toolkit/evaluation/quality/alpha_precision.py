from logging import INFO

import pandas as pd

from midst_toolkit.common.logger import log
from midst_toolkit.evaluation.metric_base import MetricBase
from midst_toolkit.evaluation.quality.synthcity.dataloader import GenericDataLoader
from midst_toolkit.evaluation.quality.synthcity.statistical_eval import (
    AlphaPrecision as SynthcityAlphaPrecision,
)


NAIVE_METRIC_SUFFIX = "naive"


class AlphaPrecision(MetricBase):
    def __init__(self, naive_only: bool = True):
        """
        Compute several quality metrics based on the Alpha Precision measure originally proposed in
        https://arxiv.org/abs/2301.07573 comparing the quality of synthetically generated data to real data. The
        implementation is based heavily on the Synthcity library (https://github.com/vanderschaarlab/synthcity).
        Specifically, this class computes the alpha-precision, beta-recall, and authenticity scores between the two
        datasets. If the ``naive_only`` boolean is True, then only the "naive" metrics are reported, i.e. metrics
        with "naive" in their name.

        Args:
            naive_only: Determines whether to report only the "naive" metrics for each of alpha-precision,
            beta-recall, and authenticity scores. Defaults to True.
        """
        self.naive_only = naive_only

    def compute(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, holdout_data: pd.DataFrame | None = None
    ) -> dict[str, float]:
        """
        Computes the alpha-precision, beta-recall, and authenticity scores between the real and synthetic datasets.
        If the ``naive_only`` boolean is True, then only the "naive" metrics are reported, i.e. metrics
        with "naive" in their name.

        NOTE: ``holdout_data`` is not used for this class.

        Args:
            real_data: Real data that the synthetic data is meant to mimic/replace.
            synthetic_data: Synthetic data to be compared against the provided real data.
            holdout_data: NOT USED for this metric.

        Returns:
            A dictionary containing the computed scores using the AlphaPrecision class in the Synthcity library.
        """
        # Wrap the dataframes in a Synthcity compatible dataloader
        real_data_loader = GenericDataLoader(real_data)
        synthetic_data_loader = GenericDataLoader(synthetic_data)

        quality_evaluator = SynthcityAlphaPrecision()
        quality_results = quality_evaluator.evaluate(real_data_loader, synthetic_data_loader)

        # Log results and filter to naive keys if requested
        for metric_key, metric_value in quality_results.items():
            log(INFO, f"{metric_key}: {metric_value}")
            if self.naive_only and (NAIVE_METRIC_SUFFIX not in metric_key):
                del quality_results[metric_key]

        return quality_results
