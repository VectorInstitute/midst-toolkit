import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.trial import FrozenTrial, Trial
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from midst_toolkit.attacks.ensemble.train_utils import get_tpr_at_fpr


optuna.logging.set_verbosity(optuna.logging.WARNING)


class XGBoostHyperparameterTuner:
    """Class for tuning XGBoost hyperparameters using Optuna."""

    def __init__(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
        use_gpu: bool = False,
    ):
        """
        Initializes the tuner with data and column information.


        Args:
            x: Input features as a DataFrame.
            y: Target variable as a numpy array.
            use_gpu: Whether to use GPU acceleration. Defaults to False.
        """
        self.x = x
        self.y = y
        self.use_gpu = use_gpu

    def _create_preprocessing_pipeline(self) -> ColumnTransformer:
        """
        Creates a preprocessing pipeline for the input features.
        It only scales continuous features using StandardScaler.

        Returns:
            A ColumnTransformer for preprocessing.
        """
        return ColumnTransformer(
            [
                ("continuous", StandardScaler(), self.x.columns),  # All features are continuous
            ],
            verbose_feature_names_out=False,
            remainder="passthrough",
        )

    def _create_xgb_pipeline(self, trial: Trial | FrozenTrial) -> Pipeline:
        """
        Creates a XGBoost pipeline for an Optuna trial.

        Args:
            trial: An Optuna trial object, which can either be dynamic or immutable.

        Returns:
            A Scikit-learn Pipeline with preprocessing and XGBoost classifier.
        """
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
                        tree_method="auto",
                        # if not self.use_gpu else "gpu_hist",
                        objective="binary:logistic",
                        seed=np.random.randint(1000),
                        verbosity=1,
                    ),
                ),
            ]
        )

    def _evaluate_pipeline_cv(self, trial: Trial, num_kfolds: int) -> float:
        """
        Performs cross-validation on the pipeline and returns the mean TPR at a fixed FPR.

        Args:
            trial: An Optuna trial object.
            num_kfolds: Number of folds for cross-validation.

        Returns:
            Mean TPR at the specified FPR across all folds.
        """
        pipeline = self._create_xgb_pipeline(trial)
        tpr_scorer = make_scorer(get_tpr_at_fpr)

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
        Performs hyperparameter tuning using Optuna and returns the best pipeline.

        Args:
            num_optuna_trials: Number of Optuna trials for hyperparameter optimization. Defaults to 50.
            num_kfolds: Number of folds for cross-validation. Defaults to 5.

        Returns:
            The best Scikit-learn Pipeline with optimized hyperparameters.

        """

        def objective(trial: Trial) -> float:
            return self._evaluate_pipeline_cv(trial, num_kfolds=num_kfolds)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(n_startup_trials=10, seed=np.random.randint(1000)),
        )
        study.optimize(objective, n_trials=num_optuna_trials)

        best_pipe = self._create_xgb_pipeline(study.best_trial)
        best_pipe.fit(self.x, self.y)

        return best_pipe
