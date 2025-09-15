# Train meta-classifier for blending++ ensemble attack

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# from train_utils import fit_lr_pipeline, hyperparam_tuning
from midst_toolkit.attacks.ensemble.XGBoost import XGBoostHyperparameterTuner


def train_meta_classifier(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    cat_cols: list,
    cont_cols: list,
    bounds: dict,
    model_type: str,
    use_gpu: bool = True,
    epochs: int = 1,
) -> Pipeline:
    """
    Trains a meta-classifier (LR or XGBoost) on the combined feature set.

    Args:
        x_train: The training features for the meta-classifier.
        y_train: The training labels.
        cat_cols: List of categorical column names.
        cont_cols: List of continuous column names.
        model_type: The type of classifier to train ('lr' or 'xgb').
        use_gpu: Whether to use GPU for XGBoost hyperparameter tuning.
        epochs: Number of epochs for training.

    Returns:
        The trained classifier pipeline.
    """
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

        tuner = XGBoostHyperparameterTuner(
            x=x_train,
            y=y_train,
            continuous_cols=cont_cols,
            categorical_cols=cat_cols,
            # bounds=bounds,
            use_gpu=True,
        )

        # Run the tuning process
        best_model = tuner.tune_hyperparameters(
            num_optuna_trials=100,
            num_kfolds=5,
        )

        # meta_classifier = hyperparam_tuning(
        #     x=x_train,
        #     y=y_train,
        #     continuous_cols=continuous_cols,
        #     categorical_cols=[],
        #     bounds={},
        #     num_optuna_trials=1000,
        #     num_kfolds=5,
        #     use_gpu=use_gpu,
        # )
    else:
        raise ValueError(f"Unsupported meta_classifier_type: {model_type}")

    return meta_classifier
