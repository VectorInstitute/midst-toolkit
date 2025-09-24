"""Blending++ orchestrator, equivalent to blending_plus_plus.py in the submission repo."""

from enum import Enum

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression

from midst_toolkit.attacks.ensemble.distance_features import calculate_domias_score, calculate_gower_features
from midst_toolkit.attacks.ensemble.train_utils import get_tpr_at_fpr
from midst_toolkit.attacks.ensemble.XGBoost import XGBoostHyperparameterTuner


class MetaClassifierType(Enum):
    LR = "lr"
    XGB = "xgb"


class BlendingPlusPlus:
    """Blending++ attack implementation."""

    def __init__(self, data_configs: DictConfig, meta_classifier_type: MetaClassifierType = MetaClassifierType.XGB):
        """
        Initializes the Blending++ attack with specified data configurations and meta-classifier type.

        Args:
            data_configs: Data configuration dictionary.
            meta_classifier_type: Type of meta classifier model. Defaults to MetaClassifierType.XGB.

        """
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
        """
        Prepares meta-classifier features by combining original continuous features,
        Gower distance features, DOMIAS predictions, and RMIA signals.

        Args:
            df_input: Input dataframe (e.g., meta-classifier train or test set).
            df_synth: Synthetic dataframe.
            df_ref: Reference (real) population dataframe.
            cat_cols: Categorical column names.
            cont_cols: Continuous column names.

        Returns:
            A dataframe with the meta-classifier features.
            The shape is (num_samples, num_original_continuous + 9 + 1 + RMIA_features).

        """
        df_synth = df_synth.reset_index(drop=True)[df_input.columns]

        # 1. Get Gower distance features
        gower_features = calculate_gower_features(
            df_input=df_input, df_synthetic=df_synth, categorical_column_names=cat_cols
        )

        # 2. Get DOMIAS predictions
        domias_features = calculate_domias_score(df_input=df_input, df_synthetic=df_synth, df_reference=df_ref)

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
    ) -> None:
        """
        Trains the Blending++ meta-classifier.

        Args:
            df_train: Dataframe for training the meta-classifier.
            y_train: Labels for the training data.
            df_synth: Synthetic dataframe.
            df_ref: Reference (real) population dataframe.
            use_gpu: Whether to use GPU acceleration. Defaults to True.
            epochs: Number of training iterations. Defaults to 1.

        """
        meta_features = self._prepare_meta_features(
            df_input=df_train,
            df_synth=df_synth,
            df_ref=df_ref,
            cat_cols=self.data_configs.metadata.categorical,
            cont_cols=self.data_configs.metadata.continuous,
        )

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

    def predict(
        self,
        df_test: pd.DataFrame,
        df_synth: pd.DataFrame,
        df_ref: pd.DataFrame,
        y_test: np.ndarray,
    ) -> tuple[np.ndarray, float | None]:
        """
        Makes predictions using the trained Blending++ meta-classifier.

        Args:
            df_test: Test dataframe for prediction.
            df_synth: Synthetic dataframe.
            df_ref: Reference (real) population dataframe.
            y_test: Test labels for evaluation.

        Note: .fit() must be called before .predict().

        Returns:
            Probabilities of membership and TPR at FPR if y_test is provided.
        """
        assert self.meta_classifier_ is not None, "You must call .fit() before .predict()"

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
