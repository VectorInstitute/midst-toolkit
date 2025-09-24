""" Blending++ orchestrator, equivalent to blending_plus_plus.py in the submission repo"""

from typing import Self
from enum import Enum

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression

from midst_toolkit.attacks.ensemble.distance_features import calculate_domias, calculate_gower_features
from midst_toolkit.attacks.ensemble.train_utils import get_tpr_at_fpr
from midst_toolkit.attacks.ensemble.XGBoost import XGBoostHyperparameterTuner

class MetaClassifierType(Enum):
    LR = "lr"
    XGB = "xgb"


class BlendingPlusPlus:
    """
    Blending++ attack implementation.
    """

    def __init__(self, data_configs: DictConfig, meta_classifier_type = MetaClassifierType.XGB):
        """
        Initializes the Blending++ attack with specified data configurations and meta-classifier type.

        Args:
            data_configs: Data configuration dictionary.
            meta_classifier_type: Type of meta classifier model. Defaults to MetaClassifierType.XGB.

        """
        # if meta_classifier_type not in ["lr", "xgb"]:
        #     raise ValueError("meta_classifier_type must be 'lr' or 'xgb'")
        self.meta_classifier_type = meta_classifier_type
        self.data_configs = data_configs
        self.meta_classifier_ = None  # The trained model, underscore denotes fitted attribute

    # TODO: Add RMIA function
    def _prepare_meta_features(
        self,
        df_input: pd.DataFrame,
        df_synth: pd.DataFrame,
        df_ref: pd.DataFrame,
        cat_cols: list,
        cont_cols: list,
    ) -> pd.DataFrame:
        """Private helper to assemble distance-based features for the meta-classifier."""
        df_synth = df_synth.reset_index(drop=True)[df_input.columns]

        # 1. Get Gower distance features
        gower_features = calculate_gower_features(df_input, df_synth, cat_cols)

        # 2. Get DOMIAS predictions
        domias_features = calculate_domias(df_input=df_input, df_synth=df_synth, df_ref=df_ref)

        # 3. Get RMIA signals (placeholder)
        rmia_signals = pd.read_csv(
            "examples/ensemble_attack_example/data/attack_data/og_rmia_train_meta_pred.csv"
        )  # Placeholder for RMIA features

        continuous_features = df_input.loc[
            :, df_input.columns.isin(cont_cols)
        ]  # Continuous features from original data

        return pd.concat(
            [
                continuous_features,
                gower_features,
                domias_features,
                rmia_signals,
            ],
            axis=1,
        )

    def fit(
        self,
        df_train: pd.DataFrame,
        y_train: np.ndarray,
        df_synth: pd.DataFrame,
        df_ref: pd.DataFrame,
        use_gpu: bool = True,
        epochs: int = 1,
    ) -> Self:
        """Trains the meta-classifier using the meta_train set."""
        meta_features = self._prepare_meta_features(
            df_input=df_train,
            df_synth=df_synth,
            df_ref=df_ref,
            cat_cols=self.data_configs.metadata.categorical,
            cont_cols=self.data_configs.metadata.continuous,
        )
        # import pdb; pdb.set_trace()

        if self.meta_classifier_type == MetaClassifierType.XGB:
            tuner = XGBoostHyperparameterTuner(
                x=meta_features,
                y=y_train,
                use_gpu=use_gpu,
            )

            # Run the tuning process
            self.meta_classifier_ = tuner.tune_hyperparameters(
                num_optuna_trials=100,
                num_kfolds=5,
            )

        elif self.meta_classifier_type == MetaClassifierType.LR:
            lr_model = LogisticRegression(max_iter=1000)
            self.meta_classifier_ = lr_model.fit(meta_features, y_train)

        else:
            raise ValueError(f"Unsupported meta_classifier_type: {self.meta_classifier_type}")

        return self

    def predict(
        self,
        df_test: pd.DataFrame,
        df_synth: pd.DataFrame,
        df_ref: pd.DataFrame,
        y_test: np.ndarray,
    ) -> np.ndarray:
        """Predicts membership probability on the meta_test set."""
        if self.meta_classifier_ is None:
            raise RuntimeError("You must call .fit() before .predict()")

        df_test_features = self._prepare_meta_features(
            df_input=df_test,
            df_synth=df_synth,
            df_ref=df_ref,
            cat_cols=self.data_configs.metadata.categorical,
            cont_cols=self.data_configs.metadata.continuous,
        )

        probabilities = self.meta_classifier_.predict_proba(df_test_features)[:, 1]

        score = None

        if y_test is not None:
            score = get_tpr_at_fpr(true_membership=y_test, predictions=probabilities, max_fpr=0.1)

        return probabilities, score
