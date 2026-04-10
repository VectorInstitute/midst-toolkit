from typing import Literal

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import numpy as np

from midst_toolkit.evaluation.metrics_base import SynthEvalMetric


class MeanF1ScoreDifference(SynthEvalMetric):
    def __init__(
        self,
        categorical_columns: list[str],
        numerical_columns: list[str],
        label_column: str,
        do_preprocess: bool = False,
        folds: int = 5,
        f1_type: Literal["micro", "macro", "samples", "weighted", "binary"] = "micro",
    ):
        """
        MODIFIED VERSION: Uses ONLY Random Forest Classifier
        
        This class computes the difference in F1 scores for Random Forest classifiers 
        trained on real and synthetic data.
        """
        super().__init__(categorical_columns, numerical_columns, do_preprocess)
        assert label_column not in numerical_columns, (
            "Label column should not be included in the set of numerical columns provided"
        )
        assert label_column not in categorical_columns, (
            "Label column should not be included in the set of categorical columns provided"
        )
        self.label_column = label_column
        self.all_columns = categorical_columns + numerical_columns + [label_column]
        self.folds = folds
        self.f1_type = f1_type

    def compute(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, holdout_data: pd.DataFrame | None = None
    ) -> dict[str, float]:
        """
        Compute F1 score difference using ONLY Random Forest Classifier.
        
        Returns:
            Dictionary containing:
            - mean_f1_difference_holdout: F1 difference (synthetic - real)
            - real_f1_holdout: F1 score when trained on real data
            - synthetic_f1_holdout: F1 score when trained on synthetic data
        """
        
        if self.do_preprocess:
            if holdout_data is None:
                real_data, synthetic_data = self.preprocess(real_data, synthetic_data)
            else:
                real_data, synthetic_data, holdout_data = self.preprocess(real_data, synthetic_data, holdout_data)

        filtered_real_data = real_data[self.all_columns]
        filtered_synthetic_data = synthetic_data[self.all_columns]
        filtered_holdout_data = holdout_data[self.all_columns] if holdout_data is not None else None

        assert self.label_column in filtered_real_data.columns, (
            f"Label column: {self.label_column} must be in real_data"
        )
        assert self.label_column in filtered_synthetic_data.columns, (
            f"Label column: {self.label_column} must be in synthetic_data"
        )

        if holdout_data is None:
            raise ValueError("Holdout data is required for F1 score computation")

        # Prepare data
        X_real = filtered_real_data.drop(columns=[self.label_column])
        y_real = filtered_real_data[self.label_column]
        X_synth = filtered_synthetic_data.drop(columns=[self.label_column])
        y_synth = filtered_synthetic_data[self.label_column]
        X_holdout = filtered_holdout_data.drop(columns=[self.label_column])
        y_holdout = filtered_holdout_data[self.label_column]
        
        # Train Random Forest on real data, test on holdout
        clf_real = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        clf_real.fit(X_real, y_real)
        y_pred_real = clf_real.predict(X_holdout)
        real_f1 = f1_score(y_holdout, y_pred_real, average=self.f1_type)
        
        # Train Random Forest on synthetic data, test on holdout
        clf_synth = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        clf_synth.fit(X_synth, y_synth)
        y_pred_synth = clf_synth.predict(X_holdout)
        synthetic_f1 = f1_score(y_holdout, y_pred_synth, average=self.f1_type)
        
        # Calculate difference
        f1_difference = synthetic_f1 - real_f1
        
        print(f"DEBUG: Real F1 Score (holdout): {real_f1:.4f}")
        print(f"DEBUG: Synthetic F1 Score (holdout): {synthetic_f1:.4f}")
        print(f"DEBUG: F1 Difference (synthetic - real): {f1_difference:.4f}")
        
        return {
            "random_forest_f1_difference_holdout": f1_difference,
            "mean_f1_difference_holdout": f1_difference,
            "real_f1_holdout": real_f1,
            "synthetic_f1_holdout": synthetic_f1,
            "f1_difference_standard_error_holdout": 0.0,  # No variance with single model
        }