# Train meta-classifier for blending++ ensemble attack

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# from train_utils import fit_lr_pipeline
from midst_toolkit.attacks.ensemble.XGBoost import XGBoostHyperparameterTuner

def train_meta_classifier(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    model_type: str,
    use_gpu: bool = True,
    epochs: int = 1,
) -> Pipeline:
    """
    Trains a meta-classifier (XGBoost or LR) on the combined feature set.

    Args:
        x_train: The training features for the meta-classifier, which is continuous only.
        y_train: The training labels.
        # cat_cols: List of categorical column names.
        # cont_cols: List of continuous column names.
        # bounds: Dictionary with possible values for each categorical column.
        model_type: The type of classifier to train ('xgb' or 'lr').
        use_gpu: Whether to use GPU for XGBoost hyperparameter tuning.
        epochs: Number of epochs for training.

    Returns:
        The trained classifier pipeline.
    """
    meta_classifier = None

    if model_type == "xgb":
        print("Training XGBoost meta-classifier...")

        tuner = XGBoostHyperparameterTuner(
            x=x_train,
            y=y_train,
            use_gpu=use_gpu,
        )

        # Run the tuning process
        meta_classifier = tuner.tune_hyperparameters(
            num_optuna_trials=2,
            num_kfolds=5,
        )

    elif model_type == "lr":
        print("Training Logistic Regression meta-classifier...")
        # meta_classifier = fit_lr_pipeline(
        #     x=x_train,
        #     y=y_train,
        #     continuous_cols=continuous_cols,
        #     categorical_cols=[],
        #     bounds={},
        # )
    else:
        raise ValueError(f"Unsupported meta_classifier_type: {model_type}")

    return meta_classifier
