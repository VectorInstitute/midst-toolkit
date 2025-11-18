import json
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from midst_toolkit.attacks.ensemble.blending import BlendingPlusPlus, MetaClassifierType


MOCK_COLUMN_TYPES_CONTENT = {
    "numerical": ["numerical_col1", "numerical_col2"],
    "categorical": ["cat_col1"],
    "id_column_name": "id_col",
}

MOCK_TARGET_DATA = {
    "selected_sets": [pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})],
    "trained_results": [{"model_info": "mock_model", "synthetic_data": [5, 6]}],
}


@pytest.fixture(scope="module")
def cfg() -> DictConfig:
    with initialize(config_path="configs"):
        return compose(config_name="data_processing_config")


@pytest.fixture
def mock_config_with_json_path():
    """Provides a mock DictConfig object with the structure required by BlendingPlusPlus."""
    return DictConfig(
        {
            "metaclassifier": {
                "data_types_file_path": "/mock/path/to/data_types.json",
                "num_optuna_trials": 100,
                "num_kfolds": 5,
                "epochs": 1,
                "meta_classifier_model_name": "mock_model_name",
            }
        }
    )


@pytest.fixture
def sample_dataframes():
    """Provides sample dataframes, now including an ID column."""
    df = pd.DataFrame(
        {
            "id_col": [10, 20, 30, 40],
            "cat_col1": ["A", "B", "A", "C"],
            "numerical_col1": [1.0, 2.0, 3.0, 4.0],
            "numerical_col2": [0.1, 0.2, 0.3, 0.4],
        }
    )

    df_synth = pd.DataFrame(
        {
            "id_col": [11, 22, 33, 44],
            "cat_col1": ["A", "B", "C", "C"],
            "numerical_col1": [1.5, 2.5, 3.5, 4.5],
            "numerical_col2": [0.15, 0.25, 0.35, 0.45],
        }
    )

    df_ref = pd.DataFrame(
        {
            "id_col": [1, 2, 3, 4, 5, 6],
            "cat_col1": ["A", "B", "C", "A", "B", "C"],
            "numerical_col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "numerical_col2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
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
    @patch("builtins.open", new_callable=mock_open)
    def test_init_success(self, mock_file, mock_config_with_json_path):
        """Tests successful initialization of BlendingPlusPlus with both XGB and LR types."""
        json_content_str = json.dumps(MOCK_COLUMN_TYPES_CONTENT)
        mock_file.return_value.read.return_value = json_content_str

        bpp_xgb = BlendingPlusPlus(
            config=mock_config_with_json_path,
            shadow_data_collection=[],
            data_types_file_path=mock_config_with_json_path.metaclassifier.data_types_file_path,
            meta_classifier_type=MetaClassifierType("xgb"),
        )

        file_path = mock_config_with_json_path.metaclassifier.data_types_file_path
        mock_file.assert_called_once_with(file_path, "r")

        assert bpp_xgb.column_types == MOCK_COLUMN_TYPES_CONTENT
        assert bpp_xgb.meta_classifier_type == MetaClassifierType.XGB
        assert bpp_xgb.trained_model is None

        mock_file.reset_mock()

        bpp_lr = BlendingPlusPlus(
            config=mock_config_with_json_path,
            shadow_data_collection=[],
            data_types_file_path=mock_config_with_json_path.metaclassifier.data_types_file_path,
            meta_classifier_type=MetaClassifierType("lr"),
        )
        assert bpp_lr.meta_classifier_type == MetaClassifierType.LR
        assert bpp_lr.trained_model is None

        mock_file.assert_called_once_with(file_path, "r")

    @patch("builtins.open", new_callable=mock_open)
    def test_init_invalid_type_raises_error(self, mock_file, mock_config_with_json_path):
        """Tests that initialization with an invalid type raises a ValueError."""
        json_content_str = json.dumps(MOCK_COLUMN_TYPES_CONTENT)
        mock_file.return_value.read.return_value = json_content_str

        with pytest.raises(ValueError):
            BlendingPlusPlus(
                config=mock_config_with_json_path,
                shadow_data_collection=[],
                data_types_file_path=mock_config_with_json_path.metaclassifier.data_types_file_path,
                meta_classifier_type=MetaClassifierType("svm"),
            )

    @patch("builtins.open", new_callable=mock_open)
    @patch("midst_toolkit.attacks.ensemble.blending.calculate_gower_features")
    @patch("midst_toolkit.attacks.ensemble.blending.calculate_domias_score")
    @patch("midst_toolkit.attacks.ensemble.blending.calculate_rmia_signals")
    def test_prepare_meta_features(
        self, mock_rmia, mock_domias, mock_gower, mock_file, mock_config_with_json_path, sample_dataframes
    ):
        """Tests the _prepare_meta_features method for correct feature assembly."""
        json_content_str = json.dumps(MOCK_COLUMN_TYPES_CONTENT)
        mock_file.return_value.read.return_value = json_content_str

        mock_gower.return_value = pd.DataFrame({"gower_1": [0.1] * 4, "gower_2": [0.2] * 4})
        mock_domias.return_value = pd.DataFrame({"domias": [0.9, 0.8, 0.7, 0.6]})
        mock_rmia.return_value = pd.DataFrame({"rmia": [1, 0, 1, 0]})

        bpp = BlendingPlusPlus(
            config=mock_config_with_json_path,
            shadow_data_collection=[],
            data_types_file_path=mock_config_with_json_path.metaclassifier.data_types_file_path,
        )

        categorical_cols = MOCK_COLUMN_TYPES_CONTENT["categorical"]
        numerical_cols = MOCK_COLUMN_TYPES_CONTENT["numerical"]
        id_col_name = MOCK_COLUMN_TYPES_CONTENT["id_column_name"]
        id_col_data = sample_dataframes["df_train"][id_col_name]

        meta_features = bpp._prepare_meta_features(
            df_input=sample_dataframes["df_train"],
            df_synthetic=sample_dataframes["df_synth"],
            df_reference=sample_dataframes["df_ref"],
            id_column_data=id_col_data,
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols,
            id_column_name=id_col_name,
        )

        mock_gower.assert_called_once()
        mock_domias.assert_called_once()
        mock_rmia.assert_called_once()

        expected_columns = ["numerical_col1", "numerical_col2", "gower_1", "gower_2", "domias", "rmia"]
        assert meta_features.shape == (4, 6)
        assert all(col in meta_features.columns for col in expected_columns)
        pd.testing.assert_series_equal(
            meta_features["numerical_col1"], sample_dataframes["df_train"]["numerical_col1"], check_names=False
        )

    @patch("builtins.open", new_callable=mock_open)
    @patch("midst_toolkit.attacks.ensemble.blending.calculate_gower_features")
    @patch("midst_toolkit.attacks.ensemble.blending.calculate_domias_score")
    @patch("midst_toolkit.attacks.ensemble.blending.calculate_rmia_signals")
    def test_prepare_meta_features_rmia_calculation(
        self, mock_rmia, mock_domias, mock_gower, mock_file, mock_config_with_json_path, sample_dataframes
    ):
        """Tests that calculate_rmia_signals is called with the correct arguments."""
        json_content_str = json.dumps(MOCK_COLUMN_TYPES_CONTENT)
        mock_file.return_value.read.return_value = json_content_str

        # Mock the return values of other feature calculators
        mock_gower.return_value = pd.DataFrame({"gower_1": [0.1] * 4})
        mock_domias.return_value = pd.DataFrame({"domias": [0.9] * 4})
        mock_rmia.return_value = pd.DataFrame({"rmia": [1] * 4})

        attack_collection = [{"name": "attack_model_1"}]

        bpp = BlendingPlusPlus(
            config=mock_config_with_json_path,
            shadow_data_collection=attack_collection,
            data_types_file_path=mock_config_with_json_path.metaclassifier.data_types_file_path,
        )

        df_train = sample_dataframes["df_train"]
        id_col_name = MOCK_COLUMN_TYPES_CONTENT["id_column_name"]
        id_col_data = df_train[id_col_name]

        bpp._prepare_meta_features(
            df_input=df_train,
            df_synthetic=sample_dataframes["df_synth"],
            df_reference=sample_dataframes["df_ref"],
            id_column_data=id_col_data,
            categorical_cols=MOCK_COLUMN_TYPES_CONTENT["categorical"],
            numerical_cols=MOCK_COLUMN_TYPES_CONTENT["numerical"],
            id_column_name=id_col_name,
        )

        mock_rmia.assert_called_once()
        _, call_kwargs = mock_rmia.call_args

        # Verify the arguments
        pd.testing.assert_frame_equal(call_kwargs["df_input"], df_train)
        assert call_kwargs["shadow_data_collection"] == attack_collection
        assert call_kwargs["categorical_column_names"] == MOCK_COLUMN_TYPES_CONTENT["categorical"]
        assert call_kwargs["id_column_name"] == id_col_name
        pd.testing.assert_series_equal(call_kwargs["id_column_data"], id_col_data)

    @patch("builtins.open", new_callable=mock_open)
    @patch("midst_toolkit.attacks.ensemble.blending.BlendingPlusPlus._prepare_meta_features")
    @patch("midst_toolkit.attacks.ensemble.blending.LogisticRegression")
    def test_fit_logistic_regression(
        self, mock_lr, mock_prepare_features, mock_file, mock_config_with_json_path, sample_dataframes
    ):
        """Tests the fit method for the Logistic Regression model."""
        mock_file.return_value.read.return_value = json.dumps(MOCK_COLUMN_TYPES_CONTENT)

        mock_prepare_features.return_value = pd.DataFrame({"feature": np.random.rand(4)})
        mock_lr_instance = MagicMock()
        mock_lr.return_value = mock_lr_instance
        mock_lr_instance.fit.return_value = mock_lr_instance

        bpp = BlendingPlusPlus(
            config=mock_config_with_json_path,
            shadow_data_collection=[],
            data_types_file_path=mock_config_with_json_path.metaclassifier.data_types_file_path,
            meta_classifier_type=MetaClassifierType("lr"),
        )
        bpp.fit(
            df_train=sample_dataframes["df_train"],
            y_train=sample_dataframes["y_train"],
            df_target_synthetic=sample_dataframes["df_synth"],
            df_reference=sample_dataframes["df_ref"],
            id_column_data=sample_dataframes["df_train"]["id_col"],
        )

        mock_prepare_features.assert_called_once()
        mock_lr.assert_called_once_with(max_iter=1000)
        mock_lr_instance.fit.assert_called_once()
        assert bpp.trained_model is mock_lr_instance

    @patch("builtins.open", new_callable=mock_open)
    @patch("midst_toolkit.attacks.ensemble.blending.BlendingPlusPlus._prepare_meta_features")
    @patch("midst_toolkit.attacks.ensemble.blending.XgBoostHyperparameterTuner")
    def test_fit_xgboost(
        self, mock_tuner_class, mock_prepare_features, mock_file, mock_config_with_json_path, sample_dataframes
    ):
        """Tests the fit method for the XGBoost model."""
        mock_file.return_value.read.return_value = json.dumps(MOCK_COLUMN_TYPES_CONTENT)

        mock_prepare_features.return_value = pd.DataFrame({"feature": np.random.rand(4)})
        mock_tuner_instance = MagicMock()
        mock_fitted_xgb = MagicMock()
        mock_tuner_instance.tune_hyperparameters.return_value = mock_fitted_xgb
        mock_tuner_class.return_value = mock_tuner_instance

        bpp = BlendingPlusPlus(
            config=mock_config_with_json_path,
            shadow_data_collection=[],
            data_types_file_path=mock_config_with_json_path.metaclassifier.data_types_file_path,
            meta_classifier_type=MetaClassifierType("xgb"),
        )
        bpp.fit(
            df_train=sample_dataframes["df_train"],
            y_train=sample_dataframes["y_train"],
            df_target_synthetic=sample_dataframes["df_synth"],
            df_reference=sample_dataframes["df_ref"],
            id_column_data=sample_dataframes["df_train"]["id_col"],
        )

        mock_prepare_features.assert_called_once()
        mock_tuner_class.assert_called_once()
        # Assert that hyperparameters are taken from the config
        mock_tuner_instance.tune_hyperparameters.assert_called_once_with(
            num_optuna_trials=mock_config_with_json_path.metaclassifier.num_optuna_trials,
            num_kfolds=mock_config_with_json_path.metaclassifier.num_kfolds,
        )
        assert bpp.trained_model is mock_fitted_xgb

    @patch("builtins.open", new_callable=mock_open)
    def test_predict_raises_error_if_not_fit(self, mock_file, mock_config_with_json_path, sample_dataframes):
        """Tests that calling .predict() before .fit() raises a RuntimeError."""
        mock_file.return_value.read.return_value = json.dumps(MOCK_COLUMN_TYPES_CONTENT)

        bpp = BlendingPlusPlus(
            config=mock_config_with_json_path,
            shadow_data_collection=[],
            data_types_file_path=mock_config_with_json_path.metaclassifier.data_types_file_path,
        )
        with pytest.raises(AssertionError):
            bpp.predict(
                df_test=sample_dataframes["df_test"],
                df_original_synthetic=sample_dataframes["df_synth"],
                df_reference=sample_dataframes["df_ref"],
                id_column_data=sample_dataframes["df_test"]["id_col"],
                y_test=sample_dataframes["y_test"],
            )

    @patch("builtins.open", new_callable=mock_open)
    @patch("midst_toolkit.attacks.ensemble.blending.BlendingPlusPlus._prepare_meta_features")
    @patch("midst_toolkit.attacks.ensemble.blending.get_tpr_at_fpr")
    def test_predict_flow(
        self, mock_get_tpr, mock_prepare_features, mock_file, mock_config_with_json_path, sample_dataframes
    ):
        """Tests the full predict flow: feature prep, prediction, and scoring."""
        mock_file.return_value.read.return_value = json.dumps(MOCK_COLUMN_TYPES_CONTENT)

        mock_prepare_features.return_value = pd.DataFrame({"feature": np.random.rand(4)})
        mock_classifier = MagicMock()
        mock_classifier.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.05, 0.95]])
        mock_get_tpr.return_value = 0.99

        bpp = BlendingPlusPlus(
            config=mock_config_with_json_path,
            shadow_data_collection=[],
            data_types_file_path=mock_config_with_json_path.metaclassifier.data_types_file_path,
        )
        bpp.trained_model = mock_classifier

        probabilities, score = bpp.predict(
            df_test=sample_dataframes["df_test"],
            df_original_synthetic=sample_dataframes["df_synth"],
            df_reference=sample_dataframes["df_ref"],
            id_column_data=sample_dataframes["df_test"]["id_col"],
            y_test=sample_dataframes["y_test"],
        )

        mock_prepare_features.assert_called_once()
        mock_classifier.predict_proba.assert_called_once()

        expected_probabilities = np.array([0.1, 0.8, 0.4, 0.95])
        np.testing.assert_array_almost_equal(probabilities, expected_probabilities)

        mock_get_tpr.assert_called_once()
        call_args = mock_get_tpr.call_args

        np.testing.assert_array_equal(call_args.kwargs["true_membership"], sample_dataframes["y_test"])
        np.testing.assert_array_almost_equal(call_args.kwargs["predictions"], expected_probabilities)
        np.testing.assert_equal(call_args.kwargs["max_fpr"], 0.1)

        assert score == 0.99
