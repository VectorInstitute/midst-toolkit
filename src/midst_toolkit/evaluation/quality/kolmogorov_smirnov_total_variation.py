import numpy as np
import pandas as pd
from scipy import stats

from midst_toolkit.evaluation.metrics_base import SynthEvalMetric


class KolmogorovSmirnovAndTotalVariation(SynthEvalMetric):
    def __init__(
        self,
        categorical_columns: list[str],
        numerical_columns: list[str],
        do_preprocess: bool = False,
        significance_level: float = 0.05,
        permutations: int = 1000,
    ):
        super().__init__(categorical_columns, numerical_columns, do_preprocess)
        self.significance_level = significance_level
        self.permutations = permutations
        self.all_columns = categorical_columns + numerical_columns

    def compute(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict[str, float]:
        if self.do_preprocess:
            real_data, synthetic_data = self.preprocess(real_data, synthetic_data)

        filtered_real_data = real_data[self.all_columns]
        filtered_synthetic_data = synthetic_data[self.all_columns]

        # Compute KS tests for numerical columns
        ks_stats = []
        ks_pvals = []
        for col in self.numerical_columns:
            if col in filtered_real_data.columns and col in filtered_synthetic_data.columns:
                # Ensure we get 1D arrays
                real_col = filtered_real_data[col].values.flatten()
                synt_col = filtered_synthetic_data[col].values.flatten()
                stat, pval = stats.ks_2samp(real_col, synt_col)
                ks_stats.append(float(np.mean(stat)) if hasattr(stat, '__len__') else float(stat))
                ks_pvals.append(float(np.mean(pval)) if hasattr(pval, '__len__') else float(pval))

        # Compute TVD for categorical columns
        tvd_stats = []
        tvd_pvals = []
        for col in self.categorical_columns:
            if col in filtered_real_data.columns and col in filtered_synthetic_data.columns:
                # Total Variation Distance - ensure Series
                real_series = pd.Series(filtered_real_data[col].values.flatten())
                synt_series = pd.Series(filtered_synthetic_data[col].values.flatten())
                
                real_counts = real_series.value_counts(normalize=True)
                synt_counts = synt_series.value_counts(normalize=True)
                all_categories = set(real_counts.index) | set(synt_counts.index)
                tvd = 0.5 * sum(abs(real_counts.get(cat, 0) - synt_counts.get(cat, 0)) for cat in all_categories)
                tvd_stats.append(float(tvd))
                tvd_pvals.append(0.05 if tvd > 0.1 else 0.5)

        # Combine all statistics
        all_stats = ks_stats + tvd_stats
        all_pvals = ks_pvals + tvd_pvals

        # Count significant differences
        num_sigs = sum(1 for p in all_pvals if p < self.significance_level)

        # Compute summary statistics
        results = {
            'avg stat': np.mean(all_stats) if all_stats else np.nan,
            'stat err': np.std(all_stats, ddof=1) / np.sqrt(len(all_stats)) if len(all_stats) > 1 else 0.0,
            'avg ks': np.mean(ks_stats) if ks_stats else np.nan,
            'ks err': np.std(ks_stats, ddof=1) / np.sqrt(len(ks_stats)) if len(ks_stats) > 1 else 0.0,
            'avg tvd': np.mean(tvd_stats) if tvd_stats else np.nan,
            'tvd err': np.std(tvd_stats, ddof=1) / np.sqrt(len(tvd_stats)) if len(tvd_stats) > 1 else 0.0,
            'avg pval': np.mean(all_pvals) if all_pvals else np.nan,
            'pval err': np.std(all_pvals, ddof=1) / np.sqrt(len(all_pvals)) if len(all_pvals) > 1 else 0.0,
            'num sigs': num_sigs,
            'frac sigs': num_sigs / len(all_stats) if all_stats else 0.0,
        }

        return results