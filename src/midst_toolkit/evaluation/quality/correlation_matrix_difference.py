import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from syntheval.metrics.utility.metric_mixed_correlation import MixedCorrelation

from midst_toolkit.evaluation.metrics_base import SynthEvalMetric


class CorrelationMatrixDifference(SynthEvalMetric):
    def __init__(
        self,
        categorical_columns: list[str],
        numerical_columns: list[str],
        do_preprocess: bool = False,
        compute_mixed_correlations: bool = False,
        save_heatmaps: bool = True,
        output_dir: str = "correlation_heatmaps",
    ):
       
        super().__init__(categorical_columns, numerical_columns, do_preprocess)
        self.compute_mixed_correlations = compute_mixed_correlations
        self.save_heatmaps = save_heatmaps
        self.output_dir = Path(output_dir)
        if self.save_heatmaps:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_heatmap(self, matrix: np.ndarray, title: str, filename: str, column_names: list[str] = None):
        """
        Save a heatmap visualization of a correlation matrix.
        
        Args:
            matrix: The correlation matrix to visualize
            title: Title for the heatmap
            filename: Filename to save the heatmap
            column_names: Names of the columns (for axis labels)
        """
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        if column_names is not None and len(column_names) <= 50:
            # Only show labels if there aren't too many columns
            sns.heatmap(
                matrix, 
                annot=False,  # Don't annotate values (too cluttered for large matrices)
                cmap='coolwarm', 
                center=0,
                vmin=-1, vmax=1,
                square=True,
                xticklabels=column_names,
                yticklabels=column_names,
                cbar_kws={'label': 'Correlation'}
            )
            plt.xticks(rotation=90, ha='right')
            plt.yticks(rotation=0)
        else:
            # For large matrices, don't show individual labels
            sns.heatmap(
                matrix, 
                annot=False,
                cmap='coolwarm', 
                center=0,
                vmin=-1, vmax=1,
                square=True,
                cbar_kws={'label': 'Correlation'}
            )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved heatmap: {save_path}")

    def compute(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict[str, float]:
       
        return results
    
    
    
    
        
   
      

        



    
    
   

    

    def compute(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict[str, float]:
        """
        Computes the Froebenius norm of the difference between the correlation matrices.
        """
        if self.do_preprocess:
            real_data, synthetic_data = self.preprocess(real_data, synthetic_data)

        self.syntheval_metric = MixedCorrelation(
            real_data=real_data,
            synt_data=synthetic_data,
            hout_data=None,
            cat_cols=self.categorical_columns,
            num_cols=self.numerical_columns,
            do_preprocessing=False,
            verbose=False,
        )

        # Get results WITH matrices
        results = self.syntheval_metric.evaluate(
            mixed_corr=self.compute_mixed_correlations, 
            return_mats=True  # Get matrices directly
        )

        # Extract matrices - SynthEval uses 'real_cor_mat' and 'synt_cor_mat' keys
        real_corr_matrix = None
        synthetic_corr_matrix = None

        # Pattern 1: Check for 'real_cor_mat' and 'synt_cor_mat' (the actual keys)
        if 'real_cor_mat' in results and 'synt_cor_mat' in results:
            real_corr_matrix = results['real_cor_mat']
            synthetic_corr_matrix = results['synt_cor_mat']
            print("Found matrices with keys: 'real_cor_mat', 'synt_cor_mat'")

        # Pattern 2: Check if matrices are directly in results
        elif 'real_mat' in results and 'synt_mat' in results:
            real_corr_matrix = results['real_mat']
            synthetic_corr_matrix = results['synt_mat']
            print("Found matrices with keys: 'real_mat', 'synt_mat'")

        # Pattern 3: Check alternative keys
        elif 'real_corr_mat' in results and 'synt_corr_mat' in results:
            real_corr_matrix = results['real_corr_mat']
            synthetic_corr_matrix = results['synt_corr_mat']
            print("Found matrices with keys: 'real_corr_mat', 'synt_corr_mat'")

        # Final fallback: compute them ourselves
        if real_corr_matrix is None or synthetic_corr_matrix is None:
            print("WARNING: Could not extract matrices from SynthEval, computing directly...")
            real_corr_matrix, synthetic_corr_matrix = self._compute_correlation_matrices_directly(
                real_data, synthetic_data
            )

        # Convert to numpy if needed
        if isinstance(real_corr_matrix, pd.DataFrame):
            real_corr_matrix = real_corr_matrix.values
        if isinstance(synthetic_corr_matrix, pd.DataFrame):
            synthetic_corr_matrix = synthetic_corr_matrix.values

        # Now compute all metrics from the SAME matrices
        diff_matrix = real_corr_matrix - synthetic_corr_matrix

        # Compute sums
        real_corr_sum = float(np.sum(real_corr_matrix))
        synthetic_corr_sum = float(np.sum(synthetic_corr_matrix))
        corr_sum_diff = real_corr_sum - synthetic_corr_sum

        # Compute Frobenius norms
        real_corr_frobenius = float(np.linalg.norm(real_corr_matrix, 'fro'))
        synthetic_corr_frobenius = float(np.linalg.norm(synthetic_corr_matrix, 'fro'))
        diff_corr_frobenius = float(np.linalg.norm(diff_matrix, 'fro'))

        # Verify consistency with original result
        if 'corr_mat_diff' in results:
            original_diff = results['corr_mat_diff']
            if abs(original_diff - diff_corr_frobenius) > 0.001:
                print(f"WARNING: Inconsistency detected!")
                print(f"  Original corr_mat_diff: {original_diff}")
                print(f"  Recomputed diff_corr_frobenius: {diff_corr_frobenius}")
                print(f"  Difference: {abs(original_diff - diff_corr_frobenius)}")
            else:
                print(f"✓ Consistency verified: corr_mat_diff matches diff_corr_frobenius")
            # Keep the original value from SynthEval
            # results['corr_mat_diff'] stays as is
        else:
            # If not in results, add it
            results['corr_mat_diff'] = diff_corr_frobenius

        # Add dimension info (update if needed)
        results['corr_mat_dims'] = real_corr_matrix.shape[0]

        # Add all other metrics
        results['real_corr_sum'] = real_corr_sum
        results['synthetic_corr_sum'] = synthetic_corr_sum
        results['corr_sum_diff'] = corr_sum_diff
        results['real_corr_frobenius'] = real_corr_frobenius
        results['synthetic_corr_frobenius'] = synthetic_corr_frobenius
        results['diff_corr_frobenius'] = diff_corr_frobenius

        # Get column names
        if self.compute_mixed_correlations:
            column_names = self.categorical_columns + self.numerical_columns
        else:
            column_names = self.numerical_columns

        # Verify we have the right number of columns
        expected_dims = len(column_names)
        actual_dims = real_corr_matrix.shape[0]
        if expected_dims != actual_dims:
            print(f"WARNING: Expected {expected_dims} columns but got {actual_dims}")
            print(f"  Mixed correlations: {self.compute_mixed_correlations}")
            print(f"  Categorical columns: {len(self.categorical_columns)}")
            print(f"  Numerical columns: {len(self.numerical_columns)}")
        else:
            print(f"✓ Column count matches: {actual_dims} columns")

        # Save heatmaps
        if self.save_heatmaps:
            self._save_heatmap(
                real_corr_matrix,
                "Correlation Matrix - Real Data",
                "correlation_matrix_real.png",
                column_names
            )

            self._save_heatmap(
                synthetic_corr_matrix,
                "Correlation Matrix - Synthetic Data",
                "correlation_matrix_synthetic.png",
                column_names
            )

            self._save_heatmap(
                diff_matrix,
                "Correlation Matrix Difference (Real - Synthetic)",
                "correlation_matrix_difference.png",
                column_names
            )

            # Save summary
            summary_path = self.output_dir / "correlation_summary.txt"
            with open(summary_path, 'w') as f:
                f.write("CORRELATION MATRIX ANALYSIS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Computation mode: {'Mixed (categorical + numerical)' if self.compute_mixed_correlations else 'Numerical only'}\n")
                f.write(f"Matrix dimensions: {real_corr_matrix.shape}\n")
                f.write(f"Number of columns: {len(column_names)}\n")
                if len(column_names) <= 50:
                    f.write(f"Column names: {', '.join(column_names)}\n\n")
                else:
                    f.write(f"(Too many columns to list)\n\n")

                f.write("MATRIX SUMS:\n")
                f.write(f"  Real correlation matrix sum: {real_corr_sum:.6f}\n")
                f.write(f"  Synthetic correlation matrix sum: {synthetic_corr_sum:.6f}\n")
                f.write(f"  Difference in sums: {corr_sum_diff:.6f}\n\n")

                f.write("FROBENIUS NORMS:\n")
                f.write(f"  Real correlation matrix Frobenius norm: {real_corr_frobenius:.6f}\n")
                f.write(f"  Synthetic correlation matrix Frobenius norm: {synthetic_corr_frobenius:.6f}\n")
                f.write(f"  Difference matrix Frobenius norm: {diff_corr_frobenius:.6f}\n")
                f.write(f"  (This should equal corr_mat_diff: {results['corr_mat_diff']:.6f})\n\n")

                f.write("ELEMENT-WISE STATISTICS:\n")
                f.write(f"  Max absolute difference: {np.max(np.abs(diff_matrix)):.6f}\n")
                f.write(f"  Mean absolute difference: {np.mean(np.abs(diff_matrix)):.6f}\n")

            print(f"Saved correlation summary: {summary_path}")

        return results



    def _compute_correlation_matrices_directly(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fallback method to compute correlation matrices directly if SynthEval doesn't return them.
        This should rarely be needed now that we know the correct keys.

        Args:
            real_data: Real data
            synthetic_data: Synthetic data

        Returns:
            Tuple of (real_corr_matrix, synthetic_corr_matrix)
        """
        print("Using fallback: computing correlations directly...")

        if self.compute_mixed_correlations:
            print("ERROR: Cannot compute mixed correlations in fallback mode.")
            print("This is a bug - SynthEval should have returned matrices.")
            raise RuntimeError("Mixed correlations require SynthEval matrices")
        else:
            # For numerical only, this is straightforward
            print("Computing numerical-only correlations...")
            real_numeric = real_data[self.numerical_columns]
            synthetic_numeric = synthetic_data[self.numerical_columns]
            real_corr_matrix = real_numeric.corr().values
            synthetic_corr_matrix = synthetic_numeric.corr().values

        return real_corr_matrix, synthetic_corr_matrix