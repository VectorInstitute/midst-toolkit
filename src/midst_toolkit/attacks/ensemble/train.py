# Train meta-classifier for blending++ ensemble attack

# FROM GEMINI:

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Assuming these are your existing utility functions
from train_utils import fit_lr_pipeline, hyperparam_tuning


def train_meta_classifier(
    x_train: pd.DataFrame, y_train: np.ndarray, model_type: str, use_gpu: bool = True
) -> Pipeline:
    """
    Trains a meta-classifier (LR or XGBoost) on the combined feature set.

    Args:
        x_train: The training features for the meta-classifier.
        y_train: The training labels.
        model_type: The type of classifier to train ('lr' or 'xgb').
        use_gpu: Whether to use GPU for XGBoost hyperparameter tuning.

    Returns:
        The trained classifier pipeline.
    """
    continuous_cols = list(x_train.columns)  # Assuming all meta-features are continuous

    if model_type == "lr":
        print("Training Logistic Regression meta-classifier...")
        meta_classifier = fit_lr_pipeline(
            x=x_train,
            y=y_train,
            continuous_cols=continuous_cols,
            categorical_cols=[],
            bounds={},
        )
    elif model_type == "xgb":
        print("Training XGBoost meta-classifier...")
        meta_classifier = hyperparam_tuning(
            x=x_train,
            y=y_train,
            continuous_cols=continuous_cols,
            categorical_cols=[],
            bounds={},
            num_optuna_trials=1000,
            num_kfolds=5,
            use_gpu=use_gpu,
        )
    else:
        raise ValueError(f"Unsupported meta_classifier_type: {model_type}")

    return meta_classifier
