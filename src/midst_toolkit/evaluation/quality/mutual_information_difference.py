import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from syntheval.metrics.utility.metric_mutual_information import MutualInformation
from sklearn.metrics import normalized_mutual_info_score

from midst_toolkit.evaluation.metrics_base import SynthEvalMetric


class MutualInformationDifference(SynthEvalMetric):
    def __init__(
        self,
        categorical_columns: list[str],
        numerical_columns: list[str],
        do_preprocess: bool = False,
        include_numerical_columns: bool = True,
        save_heatmaps: bool = True,
        output_dir: str = "mutual_information_heatmaps",
    ):
        """
        This class computes the Froebenius norm of the difference between the Mutual Information (MI) score matrices
        associated with two dataframes being compared. A smaller norm is better.

        The computation is based on:

        Ping H, Stoyanovich J, Howe B. DataSynthesizer: privacy-preserving synthetic datasets. 2017
        Presented at: Proceedings of the 29th International Conference on Scientific and Statistical Database
        Management; 2017; Chicago. [doi:10.1145/3085504.3091117]

        It leverages ``normalized_mutual_info_score`` from sklearn under the hood. The function computes the MI
        matrices, comparing the individual columns of the dataframes to each other. Then the difference of the
        two matrices is taken and the Froebenius norm computed for the final score.

        NOTE: Mutual Information works well for categorical variables. However, by default, SynthEval essentially
        just converts numerical columns to string representations for the computation. This isn't a great idea for
        things like floats. By default, this class respects SynthEval's choice, but you can override it and compute
        MI difference score for categorical columns only by setting ``include_numerical_columns`` to False, or
        providing an empty list for ``numerical_columns``.

        Args:
            categorical_columns: Column names corresponding to the categorical variables of any provided dataframe.
            numerical_columns: Column names corresponding to the numerical variables of any provided dataframe.
            do_preprocess: Whether or not to preprocess the dataframes with the default pipeline used by SynthEval.
                Defaults to False.
            include_numerical_columns: Whether to include any provided numerical columns in the MI difference score
                computation. See the note above for why you might not want to include them.
            save_heatmaps: Whether to save heatmap visualizations of the MI matrices. Defaults to True.
            output_dir: Directory where heatmaps will be saved. Defaults to "mutual_information_heatmaps".
        """
        super().__init__(categorical_columns, numerical_columns, do_preprocess)
        self.include_numerical_columns = include_numerical_columns
        self.all_columns = categorical_columns + numerical_columns
        self.save_heatmaps = save_heatmaps
        self.output_dir = Path(output_dir)
        if self.save_heatmaps:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_heatmap(self, matrix: np.ndarray, title: str, filename: str, column_names: list[str] = None):
        """
        Save a heatmap visualization of a mutual information matrix.
        
        Args:
            matrix: The MI matrix to visualize
            title: Title for the heatmap
            filename: Filename to save the heatmap
            column_names: Names of the columns (for axis labels)
        """
        plt.figure(figsize=(12, 10))
        
        # Create heatmap - MI values are between 0 and 1
        if column_names is not None and len(column_names) <= 50:
            # Only show labels if there aren't too many columns
            sns.heatmap(
                matrix, 
                annot=False,  # Don't annotate values (too cluttered for large matrices)
                cmap='viridis',  # Good for MI which is always positive
                vmin=0, vmax=1,
                square=True,
                xticklabels=column_names,
                yticklabels=column_names,
                cbar_kws={'label': 'Mutual Information'}
            )
            plt.xticks(rotation=90, ha='right')
            plt.yticks(rotation=0)
        else:
            # For large matrices, don't show individual labels
            sns.heatmap(
                matrix, 
                annot=False,
                cmap='viridis',
                vmin=0, vmax=1,
                square=True,
                cbar_kws={'label': 'Mutual Information'}
            )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved heatmap: {save_path}")

    def _save_difference_heatmap(self, matrix: np.ndarray, title: str, filename: str, column_names: list[str] = None):
        """
        Save a heatmap visualization of the MI difference matrix (can be positive or negative).
        
        Args:
            matrix: The MI difference matrix to visualize
            title: Title for the heatmap
            filename: Filename to save the heatmap
            column_names: Names of the columns (for axis labels)
        """
        plt.figure(figsize=(12, 10))
        
        # For difference matrices, center at 0 and use diverging colormap
        max_abs_val = np.max(np.abs(matrix))
        
        if column_names is not None and len(column_names) <= 50:
            sns.heatmap(
                matrix, 
                annot=False,
                cmap='coolwarm',  # Diverging colormap for differences
                center=0,
                vmin=-max_abs_val, vmax=max_abs_val,
                square=True,
                xticklabels=column_names,
                yticklabels=column_names,
                cbar_kws={'label': 'MI Difference'}
            )
            plt.xticks(rotation=90, ha='right')
            plt.yticks(rotation=0)
        else:
            sns.heatmap(
                matrix, 
                annot=False,
                cmap='coolwarm',
                center=0,
                vmin=-max_abs_val, vmax=max_abs_val,
                square=True,
                cbar_kws={'label': 'MI Difference'}
            )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved heatmap: {save_path}")

    def _compute_mi_matrix(self, data: pd.DataFrame, columns: list[str]) -> np.ndarray:
        """
        Compute mutual information matrix for a dataframe.
        
        Args:
            data: DataFrame to compute MI for
            columns: List of column names to include
            
        Returns:
            Mutual information matrix
        """
        n_cols = len(columns)
        mi_matrix = np.zeros((n_cols, n_cols))
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i == j:
                    mi_matrix[i, j] = 1.0  # Perfect MI with itself
                else:
                    # Convert to string to handle both categorical and numerical
                    col1_data = data[col1].astype(str)
                    col2_data = data[col2].astype(str)
                    mi_matrix[i, j] = normalized_mutual_info_score(col1_data, col2_data)
        
        return mi_matrix

    def compute(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict[str, float]:
        """
        Computes the Froebenius norm of the difference between the Mutual Information (MI) score matrices associated
        with  the ``real_data`` and ``synthetic_data`` dataframes. The computation is based on the work below.

        Ping H, Stoyanovich J, Howe B. DataSynthesizer: privacy-preserving synthetic datasets. 2017
        Presented at: Proceedings of the 29th International Conference on Scientific and Statistical Database
        Management; 2017; Chicago. [doi:10.1145/3085504.3091117]

        It leverages ``normalized_mutual_info_score`` from sklearn under the hood. The function computes the MI
        matrices, comparing the individual columns of the dataframes to each other. Then the difference of the
        two matrices is taken and the Froebenius norm computed for the final score.

        NOTE: Mutual Information works well for categorical variables. However, by default, SynthEval essentially
        just converts numerical columns to string representations for the computation. This isn't a great idea for
        things like floats. By default, this class respects SynthEval's choice, but you can override it and compute
        MI difference score for categorical columns only by setting ``self.include_numerical_columns`` to False, or
        ``self.numerical_columns`` to an empty list.

        Args:
            real_data: Real data to which the synthetic data may be compared. In many cases this will be data used
                to TRAIN the model that generated the synthetic data, but not always.
            synthetic_data: Synthetically generated data whose quality is to be assessed.

        Returns:
            The Froebenius norm of the difference between the two real and synthetic data MI matrices and the
            number of columns in the computed mutual information (rows/columns count of the correlation matrices).
            These are keyed under 'mutual_inf_diff' and 'mi_mat_dims' respectively.
            
            ENHANCED OUTPUTS:
            - 'real_mi_sum': Sum of all elements in the real data MI matrix
            - 'synthetic_mi_sum': Sum of all elements in the synthetic data MI matrix
            - 'mi_sum_diff': Difference between real and synthetic MI sums
            - Heatmaps saved to output_dir (if save_heatmaps=True):
                * mi_matrix_real.png
                * mi_matrix_synthetic.png
                * mi_matrix_difference.png
        """
        if self.do_preprocess:
            real_data, synthetic_data = self.preprocess(real_data, synthetic_data)

        # NOTE: The SynthEval MutualInformation class ignores column specifications by default. However, for
        # other classes (correlation_matrix_difference for example), specifying less than all of the columns restricts
        # the score computation to just those columns. To make this consistent we do that here, before passing to the
        # SynthEval class.
        filtered_real_data = (
            real_data[self.all_columns] if self.include_numerical_columns else real_data[self.categorical_columns]
        )
        filtered_synthetic_data = (
            synthetic_data[self.all_columns]
            if self.include_numerical_columns
            else synthetic_data[self.categorical_columns]
        )

        self.syntheval_metric = MutualInformation(
            real_data=filtered_real_data,
            synt_data=filtered_synthetic_data,
            hout_data=None,
            cat_cols=self.categorical_columns,
            num_cols=self.numerical_columns,
            do_preprocessing=False,
            verbose=False,
        )

        # Get the original results (Frobenius norm)
        results = self.syntheval_metric.evaluate()
        
        # Get column names for computation
        column_names = (
            self.all_columns if self.include_numerical_columns else self.categorical_columns
        )
        
        # Compute MI matrices directly (since SynthEval may not expose them)
        print("Computing MI matrices...")
        real_mi_matrix = self._compute_mi_matrix(filtered_real_data, column_names)
        synthetic_mi_matrix = self._compute_mi_matrix(filtered_synthetic_data, column_names)
        diff_matrix = real_mi_matrix - synthetic_mi_matrix
        
        # Compute sums of all elements in each matrix
        real_mi_sum = float(np.sum(real_mi_matrix))
        synthetic_mi_sum = float(np.sum(synthetic_mi_matrix))
        mi_sum_diff = real_mi_sum - synthetic_mi_sum
        
        # Compute Frobenius norms of individual matrices
        real_mi_frobenius = float(np.linalg.norm(real_mi_matrix, 'fro'))
        synthetic_mi_frobenius = float(np.linalg.norm(synthetic_mi_matrix, 'fro'))
        diff_mi_frobenius = float(np.linalg.norm(diff_matrix, 'fro'))
        
        # Add new metrics to results
        results['real_mi_sum'] = real_mi_sum
        results['synthetic_mi_sum'] = synthetic_mi_sum
        results['mi_sum_diff'] = mi_sum_diff
        results['real_mi_frobenius'] = real_mi_frobenius
        results['synthetic_mi_frobenius'] = synthetic_mi_frobenius
        results['diff_mi_frobenius'] = diff_mi_frobenius
        
        # Save heatmaps
        if self.save_heatmaps:
            self._save_heatmap(
                real_mi_matrix,
                "Mutual Information Matrix - Real Data",
                "mi_matrix_real.png",
                column_names
            )
            
            self._save_heatmap(
                synthetic_mi_matrix,
                "Mutual Information Matrix - Synthetic Data",
                "mi_matrix_synthetic.png",
                column_names
            )
            
            self._save_difference_heatmap(
                diff_matrix,
                "MI Matrix Difference (Real - Synthetic)",
                "mi_matrix_difference.png",
                column_names
            )
            
            # Also save a summary statistics file
            summary_path = self.output_dir / "mi_summary.txt"
            with open(summary_path, 'w') as f:
                f.write("MUTUAL INFORMATION MATRIX ANALYSIS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Matrix dimensions: {real_mi_matrix.shape}\n")
                f.write(f"Number of columns: {len(column_names)}\n")
                f.write(f"Column names: {', '.join(column_names)}\n")
                f.write(f"Include numerical columns: {self.include_numerical_columns}\n\n")
                
                f.write("MATRIX SUMS:\n")
                f.write(f"  Real MI matrix sum: {real_mi_sum:.6f}\n")
                f.write(f"  Synthetic MI matrix sum: {synthetic_mi_sum:.6f}\n")
                f.write(f"  Difference in sums: {mi_sum_diff:.6f}\n\n")
                
                f.write("FROBENIUS NORMS:\n")
                f.write(f"  Real MI matrix Frobenius norm: {real_mi_frobenius:.6f}\n")
                f.write(f"  Synthetic MI matrix Frobenius norm: {synthetic_mi_frobenius:.6f}\n")
                f.write(f"  Difference matrix Frobenius norm: {diff_mi_frobenius:.6f}\n")
                f.write(f"  Frobenius norm (from original metric): {results['mutual_inf_diff']:.6f}\n\n")
                
                f.write("ELEMENT-WISE STATISTICS:\n")
                f.write(f"  Max absolute difference: {np.max(np.abs(diff_matrix)):.6f}\n")
                f.write(f"  Mean absolute difference: {np.mean(np.abs(diff_matrix)):.6f}\n")
            
            print(f"Saved MI summary: {summary_path}")

        return results