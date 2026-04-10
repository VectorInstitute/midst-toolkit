from typing import Literal

import pandas as pd

# NOTE: Despite the naming convention of the SynthEval metrics class, the metric is not accuracy but rather F1
# score (of some kind).
from syntheval.metrics.utility.metric_accuracy_difference import ClassificationAccuracy as SynthEvalF1ScoreDifference

from midst_toolkit.evaluation.metrics_base import SynthEvalMetric


def _extract_from_dictionary(
    result: dict[str, float | dict[str, float]],
    upper_key: str,
    lower_key: str | None = None,
) -> float:

    value = result[upper_key]
    if lower_key is not None:
        assert isinstance(value, dict)
        return value[lower_key]
    assert isinstance(value, float)
    return value



def _post_process_results(
    result: dict[str, float | dict[str, float]], process_holdout: bool = False
) -> dict[str, float]:
    """
    Extract ONLY Random Forest results (index 0 in diffs array).
    Syntheval 1.5.0 returns arrays: [rf, adaboost, svm, logreg]
    """
    # Extract Random Forest from diffs array (index 0)
    diffs = result.get('diffs', [0, 0, 0, 0])
    
    flat_result = {
        "random_forest_f1_difference": diffs[0],  # RF is first in array
        "mean_f1_difference": result.get('avg diff', 0.0),
        "f1_difference_standard_error": result.get('avg diff err', 0.0),
    }
    
    if process_holdout:
        diffs_hout = result.get('diffs hout', [0, 0, 0, 0])
        flat_result["random_forest_f1_difference_holdout"] = diffs_hout[0]  # RF is first
        flat_result["mean_f1_difference_holdout"] = result.get('avg diff hout', 0.0)
        flat_result["f1_difference_standard_error_holdout"] = result.get('avg diff err hout', 0.0)
    
    return flat_result

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
       
        super().__init__(categorical_columns, numerical_columns, do_preprocess)
        assert label_column not in numerical_columns, (
            "Label column should not be included in the set of numerical columns provided"
        )
        assert label_column not in categorical_columns, (
            "Label column should not be included in the set of numerical columns provided"
        )
        self.label_column = label_column
        self.all_columns = categorical_columns + numerical_columns + [label_column]
        self.folds = folds
        self.f1_type = f1_type

    def compute(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, holdout_data: pd.DataFrame | None = None
    ) -> dict[str, float]:
        
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

        self.syntheval_metric = SynthEvalF1ScoreDifference(
            real_data=filtered_real_data,
            synt_data=filtered_synthetic_data,
            hout_data=filtered_holdout_data,
            # SynthEval wants cat_cols to have the analysis target (label) included so we jam it in.
            cat_cols=self.categorical_columns + [self.label_column],
            num_cols=self.numerical_columns,
            do_preprocessing=False,
            verbose=False,
            analysis_target=self.label_column,
        )
        # result = self.syntheval_metric.evaluate(F1_type=self.f1_type, k_folds=self.folds, full_output=False)
        #return _post_process_results(result, process_holdout=(holdout_data is not None))
        result = self.syntheval_metric.evaluate(F1_type=self.f1_type, k_folds=self.folds)
        print(f"DEBUG: Result keys: {result.keys()}")
        print(f"DEBUG: Full result: {result}")
        return _post_process_results(result, process_holdout=(holdout_data is not None))
