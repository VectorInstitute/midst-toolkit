from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

# The class to be tested
from midst_toolkit.attacks.ensemble.blending import BlendingPlusPlus, MetaClassifierType



# --- Fixtures: Reusable setup code for tests ---


@pytest.fixture(scope="module")
def cfg() -> DictConfig:
    with initialize(config_path="."):
        return compose(config_name="test_config")


@pytest.fixture
def mock_data_configs():
    """Provides a mock DictConfig object for tests."""
    return DictConfig({"metadata": {"categorical": ["cat_col1"], "continuous": ["cont_col1", "cont_col2"]}})


@pytest.fixture
def sample_dataframes():
    """Provides sample pandas DataFrames for testing."""
    df = pd.DataFrame(
        {
            "cat_col1": ["A", "B", "A", "C"],
            "cont_col1": [1.0, 2.0, 3.0, 4.0],
            "cont_col2": [0.1, 0.2, 0.3, 0.4],
        }
    )

    df_synth = pd.DataFrame(
        {
            "cat_col1": ["A", "B", "C", "C"],
            "cont_col1": [1.5, 2.5, 3.5, 4.5],
            "cont_col2": [0.15, 0.25, 0.35, 0.45],
        }
    )

    df_ref = pd.DataFrame(
        {
            "cat_col1": ["A", "B", "C", "A", "B", "C"],
            "cont_col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "cont_col2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )

    y = np.array([0, 1, 1, 0])

    return {
        "df_train": df,
        "y_train": y,
        "df_test": df.copy(),
        "y_test": y.copy(),
        "df_synth": df_synth,
        "df_ref": df_ref,
    }


class TestBlendingPlusPlus:
    """Groups all tests for the BlendingPlusPlus class."""

    ## Test __init__ ##
    # ------------------

    def test_init_success(self, mock_data_configs):
        """Tests successful initialization with valid meta-classifier types."""
        # Test with XGBoost
        bpp_xgb = BlendingPlusPlus(data_configs=mock_data_configs, meta_classifier_type=MetaClassifierType("xgb"))
        assert bpp_xgb.meta_classifier_type == MetaClassifierType.XGB
        assert bpp_xgb.data_configs == mock_data_configs
        assert bpp_xgb.meta_classifier_ is None

        # Test with Logistic Regression
        bpp_lr = BlendingPlusPlus(data_configs=mock_data_configs, meta_classifier_type=MetaClassifierType("lr"))
        assert bpp_lr.meta_classifier_type == MetaClassifierType.LR
        assert bpp_xgb.data_configs == mock_data_configs
        assert bpp_xgb.meta_classifier_ is None

    def test_init_invalid_type_raises_error(self, mock_data_configs):
        """Tests that initialization with an invalid type raises a ValueError."""
        with pytest.raises(ValueError):
            BlendingPlusPlus(data_configs=mock_data_configs, meta_classifier_type=MetaClassifierType("svm"))

    ## Test _prepare_meta_features ##
    # -------------------------------

    # We patch all external dependencies to isolate the method's logic
    @patch("midst_toolkit.attacks.ensemble.blending.calculate_gower_features")
    @patch("midst_toolkit.attacks.ensemble.blending.calculate_domias_score")
    @patch("pandas.read_csv")
    def test_prepare_meta_features(self, mock_read_csv, mock_domias, mock_gower, mock_data_configs, sample_dataframes):
        """Tests that _prepare_meta_features correctly calls dependencies and concatenates their outputs."""
        # 1. Setup mock return values for the patched functions
        mock_gower.return_value = pd.DataFrame({"gower_1": [0.1] * 4, "gower_2": [0.2] * 4})
        mock_domias.return_value = pd.DataFrame({"domias": [0.9, 0.8, 0.7, 0.6]})
        mock_read_csv.return_value = pd.DataFrame({"rmia": [1, 0, 1, 0]})

        # 2. Instantiate and call the method
        bpp = BlendingPlusPlus(data_configs=mock_data_configs)
        meta_features = bpp._prepare_meta_features(
            df_input=sample_dataframes["df_train"],
            df_synth=sample_dataframes["df_synth"],
            df_ref=sample_dataframes["df_ref"],
            cat_cols=mock_data_configs.metadata.categorical,
            cont_cols=mock_data_configs.metadata.continuous,
        )

        # 3. Assertions
        # Assert that our external functions were called once
        mock_gower.assert_called_once()
        mock_domias.assert_called_once()
        mock_read_csv.assert_called_once()

        # Assert the final DataFrame has the correct shape and columns
        expected_columns = ["cont_col1", "cont_col2", "gower_1", "gower_2", "domias", "rmia"]
        assert meta_features.shape == (4, 6)
        assert all(col in meta_features.columns for col in expected_columns)
        pd.testing.assert_series_equal(
            meta_features["cont_col1"], sample_dataframes["df_train"]["cont_col1"], check_names=False
        )

    ## Test fit ##
    # ------------

    @patch("midst_toolkit.attacks.ensemble.blending.BlendingPlusPlus._prepare_meta_features")
    @patch("midst_toolkit.attacks.ensemble.blending.LogisticRegression")
    def test_fit_logistic_regression(self, mock_lr, mock_prepare_features, mock_data_configs, sample_dataframes):
        """Tests the fit method for the Logistic Regression path."""
        # 1. Setup
        mock_prepare_features.return_value = pd.DataFrame({"feature": np.random.rand(4)})
        mock_lr_instance = MagicMock()
        mock_lr.return_value = mock_lr_instance

        # Configure the mock instance's 'fit' method to return the instance itself
        mock_lr_instance.fit.return_value = mock_lr_instance

        # 2. Instantiate and fit
        bpp = BlendingPlusPlus(data_configs=mock_data_configs, meta_classifier_type=MetaClassifierType("lr"))
        bpp.fit(
            df_train=sample_dataframes["df_train"],
            y_train=sample_dataframes["y_train"],
            df_synth=sample_dataframes["df_synth"],
            df_ref=sample_dataframes["df_ref"],
        )

        # 3. Assertions
        mock_prepare_features.assert_called_once()
        mock_lr.assert_called_once_with(max_iter=1000)
        mock_lr_instance.fit.assert_called_once()

        # Check that the fitted model is stored correctly
        assert bpp.meta_classifier_ is mock_lr_instance

    @patch("midst_toolkit.attacks.ensemble.blending.BlendingPlusPlus._prepare_meta_features")
    @patch("midst_toolkit.attacks.ensemble.blending.XGBoostHyperparameterTuner")
    def test_fit_xgboost(self, mock_tuner_class, mock_prepare_features, mock_data_configs, sample_dataframes):
        """Tests the fit method for the XGBoost path."""
        # 1. Setup
        mock_prepare_features.return_value = pd.DataFrame({"feature": np.random.rand(4)})
        mock_tuner_instance = MagicMock()
        mock_fitted_xgb = MagicMock()  # This represents the final, trained model
        mock_tuner_instance.tune_hyperparameters.return_value = mock_fitted_xgb
        mock_tuner_class.return_value = mock_tuner_instance

        # 2. Instantiate and fit
        bpp = BlendingPlusPlus(data_configs=mock_data_configs, meta_classifier_type=MetaClassifierType("xgb"))
        bpp.fit(
            df_train=sample_dataframes["df_train"],
            y_train=sample_dataframes["y_train"],
            df_synth=sample_dataframes["df_synth"],
            df_ref=sample_dataframes["df_ref"],
        )

        # 3. Assertions
        mock_prepare_features.assert_called_once()
        mock_tuner_class.assert_called_once()  # Check if tuner was initialized
        mock_tuner_instance.tune_hyperparameters.assert_called_once_with(num_optuna_trials=100, num_kfolds=5)
        assert bpp.meta_classifier_ is mock_fitted_xgb

    ## Test predict ##
    # ----------------

    def test_predict_raises_error_if_not_fit(self, mock_data_configs, sample_dataframes):
        """Tests that calling .predict() before .fit() raises a RuntimeError."""
        bpp = BlendingPlusPlus(data_configs=mock_data_configs)
        with pytest.raises(AssertionError):
            bpp.predict(
                df_test=sample_dataframes["df_test"],
                df_synth=sample_dataframes["df_synth"],
                df_ref=sample_dataframes["df_ref"],
                y_test=sample_dataframes["y_test"],
            )

    @patch("midst_toolkit.attacks.ensemble.blending.BlendingPlusPlus._prepare_meta_features")
    @patch("midst_toolkit.attacks.ensemble.blending.get_tpr_at_fpr")
    def test_predict_flow(self, mock_get_tpr, mock_prepare_features, mock_data_configs, sample_dataframes):
        """Tests the full predict flow: feature prep, prediction, and scoring."""
        # 1. Setup
        mock_prepare_features.return_value = pd.DataFrame({"feature": np.random.rand(4)})
        mock_classifier = MagicMock()
        # Mock predict_proba to return probabilities for class 0 and class 1
        mock_classifier.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.05, 0.95]])
        mock_get_tpr.return_value = 0.99  # Mock score

        # 2. Instantiate, manually set the fitted classifier, and predict
        bpp = BlendingPlusPlus(data_configs=mock_data_configs)
        bpp.meta_classifier_ = mock_classifier  # Manually "fit" the model

        probabilities, score = bpp.predict(
            df_test=sample_dataframes["df_test"],
            df_synth=sample_dataframes["df_synth"],
            df_ref=sample_dataframes["df_ref"],
            y_test=sample_dataframes["y_test"],
        )

        # 3. Assertions
        mock_prepare_features.assert_called_once()
        mock_classifier.predict_proba.assert_called_once()

        # Check that the correct probabilities (for class 1) are returned
        expected_probabilities = np.array([0.1, 0.8, 0.4, 0.95])
        np.testing.assert_array_almost_equal(probabilities, expected_probabilities)

        # Check that the scoring function was called correctly
        mock_get_tpr.assert_called_once()
        call_args = mock_get_tpr.call_args

        # Access the keyword arguments directly
        np.testing.assert_array_equal(call_args.kwargs["true_membership"], sample_dataframes["y_test"])
        np.testing.assert_array_almost_equal(call_args.kwargs["predictions"], expected_probabilities)
        np.testing.assert_equal(call_args.kwargs["max_fpr"], 0.1)  # You can also assert on other keyword arguments

        assert score == 0.99
