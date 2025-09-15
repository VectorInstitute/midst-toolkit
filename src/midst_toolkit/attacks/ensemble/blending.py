# Blending++ orchestrator, equivalent to blending_plus_plus.py in the submission repo

from typing import Self

import numpy as np
import pandas as pd

from midst_toolkit.attacks.ensemble.distance_features import calculate_domias, calculate_gower_features
from midst_toolkit.attacks.ensemble.train import train_meta_classifier


class BlendingPlusPlus:
    """
    Implements the Blending++ Membership Inference Attack.

    This class encapsulates the entire workflow:
    1. Generates features from Gower distance and DOMIAS.
    2. Assembles a meta-feature set.
    3. Trains a meta-classifier on these features.
    4. Predicts membership probability on new data.
    """

    def __init__(self, meta_classifier_type: str = "xgb"):
        """
        Initializes the Blending++ attack with specified meta-classifier type.
        """
        if meta_classifier_type not in ["lr", "xgb"]:
            raise ValueError("meta_classifier_type must be 'lr' or 'xgb'")
        self.meta_classifier_type = meta_classifier_type
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
        gower_features = calculate_gower_features(df_input, df_synth, cat_cols)  # shape = (20k, 9)

        # 2. Get DOMIAS predictions
        domias_features = calculate_domias(df_input=df_input, df_synth=df_synth, df_ref=df_ref)

        # 3. Get RMIA signals (placeholder)
        rmia_signals = pd.read_csv(
            "examples/ensemble_attack_example/data/attack_data/og_rmia_train_meta_pred.csv"
        )  # Placeholder for RMIA features

        continuous_features = df_input.loc[
            :, df_input.columns.isin(cont_cols)
        ]  # Continuous features from original data

        # 4. Combine all features
        df_meta = pd.concat(
            [
                continuous_features,
                gower_features,
                domias_features,
                rmia_signals,
            ],
            axis=1,
        )

        return df_meta

    def fit(
        self,
        df_train: pd.DataFrame,
        y_train: np.ndarray,
        df_synth: pd.DataFrame,
        df_ref: pd.DataFrame,
        cat_cols: list,
        cont_cols: list = None,
        bounds: dict = {},
        use_gpu: bool = True,
        epochs: int = 1,
    ) -> Self:
        """
        Trains the meta-classifier using the meta_train set.
        """
        print("Preparing meta-features for training...")

        meta_features = self._prepare_meta_features(
            df_input=df_train, df_synth=df_synth, df_ref=df_ref, cat_cols=cat_cols, cont_cols=cont_cols
        )

        print("Training the meta-classifier...")

        self.meta_classifier_ = train_meta_classifier(
            x_train=meta_features,
            y_train=y_train,
            cat_cols=cat_cols,
            cont_cols=cont_cols,
            bounds=bounds,
            model_type=self.meta_classifier_type,
            use_gpu=use_gpu,
            epochs=epochs,
        )

        print("Blending++ meta-classifier has been trained.")

        return self

    def predict(
        self,
        df_test: pd.DataFrame,
        df_synth: pd.DataFrame,
        df_ref: pd.DataFrame,
        cat_cols: list,
        y_test: np.ndarray,
    ) -> np.ndarray:
        """
        Predicts membership probability on the meta_test set.
        """
        if self.meta_classifier_ is None:
            raise RuntimeError("You must call .fit() before .predict()")

        print("Preparing the meta-test features for prediction...")
        df_test_features = self._prepare_meta_features(
            df_input=df_test, df_synth=df_synth, df_ref=df_ref, cat_cols=cat_cols
        )

        print("Predicting with trained meta-classifier...")
        pred_proba = self.meta_classifier_.predict_proba(df_test_features)

        # TODO: Evaluate predictions if y_test is provided

        return pred_proba
