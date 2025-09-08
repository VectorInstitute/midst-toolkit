# Blending++ orchestrator, equivalent to blending_plus_plus.py in the submission repo

import numpy as np
import pandas as pd

# from src.attack import domias
from distance_features import calculate_gower_features
from train import train_meta_classifier


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
        if meta_classifier_type not in ["lr", "xgb"]:
            raise ValueError("meta_classifier_type must be 'lr' or 'xgb'")
        self.meta_classifier_type = meta_classifier_type
        self.meta_classifier_ = None  # The trained model, underscore denotes fitted attribute

    # TODO: Add RMIA function
    def _prepare_meta_features(
        self,
        df_features: pd.DataFrame,
        df_synth: pd.DataFrame,
        df_ref: pd.DataFrame,
        cat_cols: list,
    ) -> pd.DataFrame:
        """Private helper to assemble all features for the meta-classifier."""
        # 1. Get Gower distance features
        gower_features = calculate_gower_features(df_features, df_synth, cat_cols)

        # 2. Get DOMIAS predictions
        pred_proba_domias = domias.fit_pred(df_ref=df_ref, df_synth=df_synth[df_ref.columns], df_test=df_features)
        domias_features = pd.DataFrame(pred_proba_domias, columns=["pred_proba_domias"], index=df_features.index)

        # 3. Combine all features
        df_meta = pd.concat(
            [
                df_features[config.metadata["continuous"]],  # Original continuous features
                gower_features,
                domias_features,
                pred_proba_rmia,
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
    ):
        """
        Trains the meta-classifier using the meta_train set.
        """
        print("Preparing meta-features for training...")
        df_train_meta = self._prepare_meta_features(
            df_features=df_train,
            df_synth=df_synth,
            df_ref=df_ref,
            cat_cols=cat_cols,
        )

        self.meta_classifier_ = train_meta_classifier(
            x_train=df_train_meta, y_train=y_train, model_type=self.meta_classifier_type
        )
        print("Blending++ meta-classifier has been trained.")

        # TODO: Save trained model
        return self

    def predict_proba(
        self,
        df_test: pd.DataFrame,
        df_synth: pd.DataFrame,
        df_ref: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predicts membership probability on the meta_test set.
        """
        if self.meta_classifier_ is None:
            raise RuntimeError("You must call .fit() before .predict_proba()")

        print("Preparing the test meta-features for prediction...")
        df_test_features = self._prepare_meta_features()

        print("Predicting with trained meta-classifier...")
        pred_proba = None

        return pred_proba
