"""Blending++ orchestrator, equivalent to blending_plus_plus.py in the submission repository. (https://github.com/CRCHUM-CITADEL/ensemble-mia)."""

import json
from enum import Enum

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression

from midst_toolkit.attacks.ensemble.distance_features import calculate_domias_score, calculate_gower_features
from midst_toolkit.attacks.ensemble.rmia.rmia_calculation import calculate_rmia_signals
from midst_toolkit.attacks.ensemble.train_utils import get_tpr_at_fpr
from midst_toolkit.attacks.ensemble.xgboost_tuner import XgBoostHyperparameterTuner


class MetaClassifierType(Enum):
    LR = "lr"
    XGB = "xgb"


class BlendingPlusPlus:
    def __init__(
        self,
        config: DictConfig,
        attack_data_collection: list[dict],
        target_data_collection: list[dict],
        meta_classifier_type: MetaClassifierType = MetaClassifierType.XGB,
        random_seed: int | None = None,
    ) -> None:
        """
        Initializes the Blending++ attack with specified data configurations and meta-classifier type.

        This class encapsulates the entire workflow:
        1. Generates features from Gower distance and DOMIAS.
        2. Calculates RMIA signals using the provided attack data, which contains training/fine-tuning data
            and the synthetic data generated.
        3. Assembles a meta-feature set (original numerical features + Gower + DOMIAS + RMIA).
        4. Trains a meta-classifier on these features.
        5. Predicts membership probability on new data.

        Args:
            config: Dictionary storing data configuration paths and parameters, used to load data properties.
            attack_data_collection: List of training data of the shadow models and their generated synthetic data.
            target_data_collection: List of training data of the target model and its generated synthetic data.
            meta_classifier_type: Type of meta classifier model. Defaults to MetaClassifierType.XGB.
            random_seed: Random seed for reproducibility. Defaults to None.

        """
        with open(config.data_processing_config.data_types_file_path, "r") as f:
            self.column_types = json.load(f)

        self.attack_data_collection = attack_data_collection
        self.target_data_collection = target_data_collection
        self.meta_classifier_type = meta_classifier_type
        self.trained_model = None
        self.random_seed = random_seed

    # TODO: Add RMIA function
    def _prepare_meta_features(
        self,
        df_input: pd.DataFrame,
        df_synthetic: pd.DataFrame,
        df_reference: pd.DataFrame,
        categorical_cols: list[str],
        numerical_cols: list[str],
    ) -> pd.DataFrame:
        """
        Prepares meta-classifier features by combining original continuous features,
        Gower distance features, DOMIAS predictions, and RMIA signals.

        Args:
            df_input: Input dataframe (e.g., meta-classifier train or test set).
            df_synthetic: Synthetic dataframe.
            df_reference: Real population dataframe, used as a reference for calculating the DOMIAS score.
            categorical_cols: Categorical column names.
            numerical_cols: Numerical column names.

        Returns:
            A dataframe with the meta-classifier features.
            Its shape is (num_samples, num_original_numerical + 9 + 1 + RMIA_features).

        """
        df_synthetic = df_synthetic.reset_index(drop=True)[df_input.columns]

        # 1. Get Gower distance features
        gower_features = calculate_gower_features(
            df_input=df_input, df_synthetic=df_synthetic, categorical_column_names=categorical_cols
        )

        # 2. Get DOMIAS predictions
        domias_features = calculate_domias_score(
            df_input=df_input, df_synthetic=df_synthetic, df_reference=df_reference
        )

        # 3. Get RMIA signals
        # TODO: make sure df_input has IDs (assuming shadow models and target model synth data have IDs too)
        rmia_signals = calculate_rmia_signals(
            df_input=df_input,
            attack_data_collection=self.attack_data_collection,
            target_data_collection=self.target_data_collection,
            categorical_column_names=categorical_cols,
        )

        # # (borrowed from the attack implementation repository,
        # # at https://github.com/CRCHUM-CITADEL/ensemble-mia/tree/main/input/tabddpm_black_box/meta_classifier)
        # # Will be removed after our own implementation is ready.
        # rmia_signals = pd.read_csv(
        #     "examples/ensemble_attack/data/attack_data/og_rmia_train_meta_pred.csv"
        # )  # Placeholder for RMIA features

        original_numerical_features = df_input[numerical_cols]  # Numerical features from original data

        return pd.concat(
            [
                original_numerical_features,
                gower_features,
                domias_features,
                rmia_signals,
            ],
            axis=1,
        )

    # TODO: Handle epochs parameter
    def fit(
        self,
        df_train: pd.DataFrame,
        y_train: np.ndarray,
        df_synthetic: pd.DataFrame,
        df_reference: pd.DataFrame,
        use_gpu: bool = True,
        epochs: int = 1,
    ) -> None:
        """
        Trains the Blending++ meta-classifier.

        Args:
            df_train: Dataframe for training the meta-classifier. This training set is derived from the population
                dataset which is all the data the attacker has access to (all the other attacks' training data,
                holdout data, and the challenge dataset).
                The meta training set is a combination of the "real train" data and "real control val", which is
                the data used to validate the diffusion model to generate synthetic data.
            y_train: Labels for the meta-classifier training data.
            df_synthetic: Synthetic dataframe, generated by the diffusion model.
            df_reference: Reference (real) population dataframe.
            use_gpu: Whether to use GPU acceleration. Defaults to True.
            epochs: Number of training iterations. Defaults to 1.

        """
        meta_features = self._prepare_meta_features(
            df_input=df_train,
            df_synthetic=df_synthetic,
            df_reference=df_reference,
            categorical_cols=self.column_types["categorical"],
            numerical_cols=self.column_types["numerical"],
        )

        if self.meta_classifier_type == MetaClassifierType.XGB:
            tuner = XgBoostHyperparameterTuner(
                input_features=meta_features,
                labels=y_train,
                use_gpu=use_gpu,
                random_seed=self.random_seed,
            )

            # Run the tuning process
            self.trained_model = tuner.tune_hyperparameters(
                num_optuna_trials=100,
                num_kfolds=5,
            )

        elif self.meta_classifier_type == MetaClassifierType.LR:
            lr_model = LogisticRegression(max_iter=1000)
            self.trained_model = lr_model.fit(meta_features, y_train)

        else:
            raise ValueError(f"Unsupported meta_classifier_type: {self.meta_classifier_type}")

    def predict(
        self,
        df_test: pd.DataFrame,
        df_synthetic: pd.DataFrame,
        df_reference: pd.DataFrame,
        y_test: np.ndarray,
    ) -> tuple[np.ndarray, float | None]:
        """
        Makes predictions using the trained Blending++ meta-classifier.

        Args:
            df_test: DataFrame containing the test data for prediction. In the context of the MIDST Challenge, this
                represents the challenge dataset for which results are required. For evaluating training performance,
                this can be the meta-test set, which is derived from the population dataset (combining all other
                attacks' training data, holdout data, and the challenge dataset).
                The meta-test set includes "real train" data and "real control test" data used to evaluate the
                diffusion model's synthetic data generation.
            df_synthetic: DataFrame containing synthetic data generated by the diffusion model.
            df_reference: DataFrame of the real population data, used as a reference for calculating the DOMIAS score.
            y_test: Optional array of test labels for evaluation. A label of "1" indicates membership in the
                diffusion model's training set, while "0" indicates non-membership.

        Note:
            The .fit() method must be called before invoking .predict().

        Returns:
            A tuple containing:
            - Probabilities of membership for the test data.
            - TPR at FPR (if y_test is provided), or None otherwise.
        """
        assert self.trained_model is not None, "You must call .fit() before .predict()"

        df_test_features = self._prepare_meta_features(
            df_input=df_test,
            df_synthetic=df_synthetic,
            df_reference=df_reference,
            categorical_cols=self.column_types["categorical"],
            numerical_cols=self.column_types["numerical"],
        )

        probabilities = self.trained_model.predict_proba(df_test_features)[:, 1]

        score = None

        if y_test is not None:
            score = get_tpr_at_fpr(true_membership=y_test, predictions=probabilities, max_fpr=0.1)

        return probabilities, score
