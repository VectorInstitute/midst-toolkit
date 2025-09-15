# Standard library
from typing import Callable, List, Optional

# 3rd party packages
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import make_scorer, roc_curve
import xgboost as xgb
from optuna.trial import Trial
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Local
# from .stats import get_tpr_at_fpr


class XGBoostHyperparameterTuner:
    """
    A class for performing hyperparameter tuning of an XGBoost classifier using Optuna.

    Attributes:
        x (pd.DataFrame): The input features.
        y (np.ndarray): The target variable.
        continuous_cols (List[str]): Names of continuous columns.
        categorical_cols (List[str]): Names of categorical columns.
        bounds (dict): Dictionary with categories for each categorical column.
        use_gpu (bool): Flag to enable GPU acceleration.
    """

    def __init__(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
        continuous_cols: List[str],
        categorical_cols: List[str],
        bounds: dict,
        use_gpu: bool = False,
    ):
        """
        Initializes the tuner with data and column information.
        """
        self.x = x
        self.y = y
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.bounds = bounds
        self.use_gpu = use_gpu

    def _create_preprocessing_pipeline(self) -> ColumnTransformer:
        """Creates and returns the data preprocessing pipeline."""
        return ColumnTransformer(
            [
                ("continuous", StandardScaler(), self.continuous_cols),
                (
                    "categorical",
                    OneHotEncoder(
                        categories=[self.bounds[c]["categories"] for c in self.categorical_cols],
                        handle_unknown="ignore",
                    ),
                    self.categorical_cols,
                ),
            ],
            verbose_feature_names_out=False,
            remainder="passthrough",
        )

    def _create_xgb_pipeline(self, trial: Trial) -> Pipeline:
        """Creates a XGBoost pipeline for an Optuna trial."""
        preprocessing = self._create_preprocessing_pipeline()
        return Pipeline(
            steps=[
                ("preprocessing", preprocessing),
                (
                    "xgboost",
                    xgb.XGBClassifier(
                        n_estimators=100,
                        eta=trial.suggest_float("eta", 0.0001, 0.1, log=True),
                        max_depth=trial.suggest_int("max_depth", 3, 10),
                        subsample=trial.suggest_float("subsample", 0.1, 1),
                        colsample_bytree=trial.suggest_float("colsample_bylevel", 0.5, 1),
                        reg_alpha=trial.suggest_categorical("reg_alpha", [0, 0.1, 0.5, 1, 5, 10]),
                        reg_lambda=trial.suggest_categorical("reg_lambda", [0, 0.1, 0.5, 1, 5, 10, 100]),
                        tree_method="auto" if not self.use_gpu else "gpu_hist",
                        objective="binary:logistic",
                        seed=np.random.randint(1000),
                        verbosity=1,
                    ),
                ),
            ]
        )

    def _get_tpr_at_fpr(
    true_membership: np.ndarray,
    predictions: np.ndarray,
    max_fpr: float = 0.1,
) -> float:
        """Calculates the best True Positive Rate when the False Positive Rate is at most `max_fpr`.

        :param true_membership: an array of values in {0,1} indicating the membership of each
            data point. 0: "non-member", 1: "member".
        :param predictions: an array of values in the range [0,1] indicating the confidence
                that a data point is a member.
        :param max_fpr: threshold on the FPR.

        return: The TPR at `max_fpr` FPR.
        """
        fpr, tpr, _ = roc_curve(true_membership, predictions)

        return max(tpr[fpr <= max_fpr])
    
    
    def _evaluate_pipeline_cv(self, trial: Trial, num_kfolds: int) -> float:
        """
        Runs k-fold cross-validation to evaluate the pipeline for a given trial.
        """
        pipeline = self._create_xgb_pipeline(trial)
        tpr_scorer = make_scorer(self._get_tpr_at_fpr, needs_proba=True)
        cv_scores = cross_val_score(
            pipeline,
            self.x,
            self.y,
            cv=num_kfolds,
            scoring=tpr_scorer,
        )
        return np.mean(cv_scores)
    


    def tune_hyperparameters(
        self,
        num_optuna_trials: int = 50,
        num_kfolds: int = 5,
    ) -> Pipeline:
        """
        Performs hyperparameter tuning and returns the best model.

        :param num_optuna_trials: Number of trials for the optimization.
        :param num_kfolds: Number of folds for cross-validation.
        :return: The best Scikit-learn Pipeline with optimized hyperparameters.
        """
        objective = lambda trial: self._evaluate_pipeline_cv(trial, num_kfolds=num_kfolds)
        
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(n_startup_trials=10, seed=np.random.randint(1000)),
        )
        study.optimize(objective, n_trials=num_optuna_trials)

        best_pipe = self._create_xgb_pipeline(study.best_trial)
        best_pipe.fit(self.x, self.y)
        return best_pipe