# Blending++ orchestrator, equivalent to blending_plus_plus.py in the submission repo

from typing import Self

import numpy as np
import pandas as pd

# from src.attack import domias
from midst_toolkit.attacks.ensemble.distance_features import calculate_gower_features, domias
# from midst_toolkit.attacks.ensemble.train import train_meta_classifier


# from src import config # Assuming config.metadata is available


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
    ) -> pd.DataFrame:
        """Private helper to assemble all features for the meta-classifier."""

        print("WE HERE")

        # 1. Get Gower distance features
        # gower_features = calculate_gower_features(df_input, df_synth, cat_cols)

        # 2. Get DOMIAS predictions
        pred_proba_domias = domias(df_input=df_input, df_synth=df_synth, df_ref=df_ref)
        domias_features = pd.DataFrame(pred_proba_domias, columns=["pred_proba_domias"], index=df_input.index)

        import pdb; pdb.set_trace()

        rmia_signals = pd.DataFrame(
            np.zeros((df_input.shape[0], 1)), columns=["rmia_placeholder"], index=df_input.index
        )

        # 3. Combine all features
        df_meta = pd.concat(
            [
                # df_input[config.metadata["continuous"]],  # Original continuous features
                # gower_features,
                domias_features,
                rmia_signals,  # Placeholder for RMIA features
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
        epochs: int = 1,
    ) -> Self:
        """
        Trains the meta-classifier using the meta_train set.
        """
        print("Preparing meta-features for training...")
        df_train_meta = self._prepare_meta_features(
            df_input=df_train,
            df_synth=df_synth,
            df_ref=df_ref,
            cat_cols=cat_cols,
        )

        # self.meta_classifier_ = train_meta_classifier(
        #     x_train=df_train_meta, y_train=y_train, model_type=self.meta_classifier_type, epochs=epochs
        # )

        # print("Blending++ meta-classifier has been trained.")

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
