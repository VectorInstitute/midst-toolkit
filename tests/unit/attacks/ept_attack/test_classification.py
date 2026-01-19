from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from midst_toolkit.attacks.ept.classification import (
    MLPClassifier,
    filter_data,
    get_scores,
    train_attack_classifier,
    train_mlp,
)
from midst_toolkit.common.variables import DEVICE


@pytest.fixture
def sample_features_df():
    # Sample DataFrame for testing filter_data.
    data = {
        "feature1_actual": [1, 2, 3],
        "feature1_error": [0.1, 0.2, 0.3],
        "feature2_error_ratio": [0.01, 0.02, 0.03],
        "feature3_accuracy": [0.9, 0.8, 0.7],
        "feature4_prediction": [1, 0, 1],
        "another_feature": [4, 5, 6],
    }
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    "columns_list, expected_cols",
    [
        (["actual"], ["feature1_actual", "another_feature"]),
        (["error"], ["feature1_error"]),
        (["error_ratio"], ["feature2_error_ratio"]),
        (["accuracy"], ["feature3_accuracy"]),
        (["prediction"], ["feature4_prediction"]),
        (["error", "accuracy"], ["feature1_error", "feature3_accuracy"]),
        (
            ["actual", "prediction"],
            ["feature1_actual", "another_feature", "feature4_prediction"],
        ),
        ([], []),
    ],
)
def test_filter_data(sample_features_df, columns_list, expected_cols):
    # Tests that filter_data correctly selects columns based on suffixes.
    result = filter_data(sample_features_df, columns_list)
    expected = sample_features_df[expected_cols].values
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (3, len(expected_cols))


def test_mlp_classifier():
    # Tests the MLPClassifier initialization and forward pass.
    input_size, hidden_size, output_size = 10, 5, 1

    model = MLPClassifier(input_size, hidden_size, output_size)

    # Check layer types and sizes
    assert isinstance(model.layers[0], torch.nn.Linear)
    assert model.layers[0].in_features == input_size
    assert model.layers[0].out_features == hidden_size
    assert isinstance(model.layers[2], torch.nn.Linear)
    assert model.layers[2].in_features == hidden_size
    assert model.layers[2].out_features == output_size

    # Test forward pass
    input_tensor = torch.randn(4, input_size)
    output = model(input_tensor)
    assert output.shape == (4, output_size)
    assert torch.all(output >= 0) and torch.all(output <= 1)


@patch("midst_toolkit.attacks.ept.classification.MLPClassifier")
def test_train_mlp(mock_mlp_class):
    # Tests the train_mlp function's logic and outputs.
    mock_model = MagicMock()
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]

    mock_model.return_value = torch.rand(10, 1, requires_grad=True)

    eval_output_mock = MagicMock()
    eval_output_mock.squeeze.return_value.cpu.return_value.numpy.return_value = np.array([0.6, 0.4, 0.7])

    mock_model.side_effect = [
        torch.rand(10, 1, requires_grad=True),  # 1st call (training)
        eval_output_mock,  # 2nd call (evaluation)
    ]

    mock_mlp_class.return_value.to.return_value = mock_model

    x_train = np.random.rand(10, 5)
    y_train = np.random.randint(0, 2, 10)
    x_test = np.random.rand(3, 5)

    # Test with eval=True
    y_pred, y_proba = train_mlp(x_train, y_train, x_test, DEVICE, epochs=1)
    assert y_pred is not None
    assert y_proba is not None
    assert y_pred.shape == (3,)
    assert y_proba.shape == (3,)
    np.testing.assert_array_equal(y_pred, np.array([1, 0, 1]))

    mock_model.side_effect = [
        torch.rand(10, 1, requires_grad=True),
        eval_output_mock,
    ]

    # Test with eval=False
    y_pred_no_eval, y_proba_no_eval = train_mlp(x_train, y_train, x_test=None, device=DEVICE, epochs=1)
    assert y_pred_no_eval is None
    assert y_proba_no_eval is None


def test_get_scores():
    # Tests the get_scores function with known values.
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    fpr_thresholds = [0.5, 0.1]

    scores = get_scores(y_true, y_proba, y_pred, fpr_thresholds)

    # Accuracy = (2 TP + 2 TN) / 6 = 4/6
    assert scores["accuracy"] == pytest.approx(4 / 6)

    # AUC-ROC = 2/3 for this specific arrangement
    assert scores["AUC-ROC"] == pytest.approx(2 / 3)

    # At a threshold of 0.5, FPR is 1/3 and TPR is 2/3. This is the highest TPR for FPR <= 0.5.
    assert scores["TPR at FPR 50.0"] == pytest.approx(2 / 3)

    # For FPR <= 0.1, max TPR is 1/3 (achieved at actual FPR of 0).
    assert scores["TPR at FPR 10.0"] == pytest.approx(1 / 3)


@pytest.fixture
def attack_data():
    # Sample data for training an attack classifier
    x_train = pd.DataFrame({"feature_error": np.random.rand(20)})
    y_train = pd.Series(np.random.randint(0, 2, 20))
    x_test = pd.DataFrame({"feature_error": np.random.rand(10)})
    y_test = pd.Series(np.random.randint(0, 2, 10))
    return x_train, y_train, x_test, y_test


@pytest.mark.parametrize("classifier_type", ["XGBoost", "CatBoost"])
@patch("midst_toolkit.attacks.ept.classification.XGBClassifier")
@patch("midst_toolkit.attacks.ept.classification.CatBoostClassifier")
def test_train_attack_classifier_tree_models(mock_catboost, mock_xgboost, classifier_type, attack_data):
    # Tests train_attack_classifier for XGBoost and CatBoost
    x_train, y_train, x_test, y_test = attack_data
    columns_list = ["error"]

    mock_model = MagicMock()
    mock_model.predict.return_value = np.zeros(10)
    mock_model.predict_proba.return_value = np.zeros((10, 2))
    if classifier_type == "XGBoost":
        mock_xgboost.return_value = mock_model
    else:
        mock_catboost.return_value = mock_model

    results = train_attack_classifier(classifier_type, columns_list, x_train, y_train, x_test, y_test)

    assert "prediction_results" in results
    assert "scores" in results
    assert "y_true" in results["prediction_results"]
    mock_model.fit.assert_called_once()
    mock_model.predict.assert_called_once()
    mock_model.predict_proba.assert_called_once()


def test_train_attack_classifier_mismatched_data(attack_data):
    # Tests that train_attack_classifier raises errors for mismatched data shapes
    x_train, y_train, x_test, y_test = attack_data
    columns_list = ["error"]

    # Test mismatches
    with pytest.raises(AssertionError, match="Mismatch in number of training samples and labels"):
        train_attack_classifier("XGBoost", columns_list, x_train.head(10), y_train, x_test, y_test)

    with pytest.raises(AssertionError, match="Mismatch in number of test samples and labels"):
        train_attack_classifier("XGBoost", columns_list, x_train, y_train, x_test.head(5), y_test)

    x_test_wrong_features = x_test.rename(columns={"feature_error": "another_feature"})
    with pytest.raises(AssertionError, match="Mismatch in number of features between train and test sets"):
        train_attack_classifier("XGBoost", columns_list, x_train, y_train, x_test_wrong_features, y_test)


@patch("midst_toolkit.attacks.ept.classification.train_mlp")
def test_train_attack_classifier_mlp(mock_train_mlp, attack_data):
    # Tests train_attack_classifier for the MLP model
    x_train, y_train, x_test, y_test = attack_data
    columns_list = ["error"]
    mock_train_mlp.return_value = (np.zeros(10), np.zeros(10))

    results = train_attack_classifier("MLP", columns_list, x_train, y_train, x_test, y_test)

    assert "prediction_results" in results
    assert "scores" in results
    mock_train_mlp.assert_called_once()

    assert mock_train_mlp.call_args[1]["eval"] is True


def test_train_attack_classifier_unsupported(attack_data):
    # Tests that an unsupported classifier type raises an assertion error
    x_train, y_train, x_test, y_test = attack_data
    with pytest.raises(AssertionError, match="Unsupported classifier type: SVM"):
        train_attack_classifier("SVM", ["error"], x_train, y_train, x_test, y_test)
