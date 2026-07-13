from enum import Enum
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from midst_toolkit.attacks.ept.classification import (
    ClassifierType,
    ColumnType,
    MLPClassifier,
    filter_data,
    get_scores,
    train_attack_classifier,
)
from midst_toolkit.common.variables import DEVICE


class MockClassifierType(Enum):
    UNSUPPORTED = "Unsupported"


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
    "column_types, expected_cols",
    [
        pytest.param(
            [ColumnType.ACTUAL],
            ["feature1_actual", "another_feature"],
            id="filter_actual_columns",
        ),
        pytest.param(
            [ColumnType.ERROR],
            ["feature1_error"],
            id="filter_error_columns",
        ),
        pytest.param(
            [ColumnType.ERROR_RATIO],
            ["feature2_error_ratio"],
            id="filter_error_ratio_columns",
        ),
        pytest.param(
            [ColumnType.ACCURACY],
            ["feature3_accuracy"],
            id="filter_accuracy_columns",
        ),
        pytest.param(
            [ColumnType.PREDICTION],
            ["feature4_prediction"],
            id="filter_prediction_columns",
        ),
        pytest.param(
            [ColumnType.ERROR, ColumnType.ACCURACY],
            ["feature1_error", "feature3_accuracy"],
            id="filter_multiple_column_types",
        ),
        pytest.param(
            [ColumnType.ACTUAL, ColumnType.PREDICTION],
            ["feature1_actual", "feature4_prediction", "another_feature"],
            id="filter_actual_and_prediction_columns",
        ),
        pytest.param(
            [],
            [],
            id="filter_no_columns_empty_list",
        ),
    ],
)
def test_filter_data(sample_features_df, column_types, expected_cols):
    """
    Test that filter_data correctly selects columns based on ColumnType suffixes.

    Verifies that:
    - The correct columns are selected for each ColumnType
    - Multiple ColumnTypes can be filtered simultaneously
    - Empty column_types list returns empty result
    - Output shape matches expected dimensions
    - Values are preserved correctly
    """
    # Act
    result = filter_data(sample_features_df, column_types)

    # Assert
    expected = sample_features_df[expected_cols].values if expected_cols else np.array([]).reshape(3, 0)

    np.testing.assert_array_equal(
        result,
        expected,
        err_msg=f"Filtered data does not match expected for column_types: {column_types}",
    )
    assert result.shape == (3, len(expected_cols)), (
        f"Result shape {result.shape} does not match expected shape (3, {len(expected_cols)})"
    )


def test_mlp_classifier():
    """
    Test the MLPClassifier initialization and forward pass.

    Verifies that:
    - Model layers are initialized with correct dimensions
    - Forward pass produces correct output shape
    - Output values are in valid range [0, 1] (sigmoid activation)
    """
    # Arrange
    input_size, hidden_size, output_size = 10, 5, 1
    batch_size = 4

    # Act
    model = MLPClassifier(input_size, hidden_size, output_size)

    # Assert - Check layer types and sizes
    assert isinstance(model.layers[0], torch.nn.Linear), "First layer should be Linear"
    assert model.layers[0].in_features == input_size, f"Expected input features: {input_size}"
    assert model.layers[0].out_features == hidden_size, f"Expected hidden features: {hidden_size}"
    assert isinstance(model.layers[2], torch.nn.Linear), "Third layer should be Linear"
    assert model.layers[2].in_features == hidden_size, f"Expected hidden to output features: {hidden_size}"
    assert model.layers[2].out_features == output_size, f"Expected output features: {output_size}"

    # Test forward pass
    input_tensor = torch.randn(batch_size, input_size)
    output = model(input_tensor)

    # The output may be squeezed, so check for either (batch_size,) or (batch_size, output_size)
    assert output.shape in [(batch_size,), (batch_size, output_size)], (
        f"Output shape {output.shape} should be either ({batch_size},) or ({batch_size}, {output_size})"
    )
    assert torch.all(output >= 0) and torch.all(output <= 1), (
        "Output values should be in range [0, 1] due to sigmoid activation"
    )


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


@pytest.mark.parametrize("classifier_type", [ClassifierType.XGBOOST, ClassifierType.CATBOOST])
@patch("midst_toolkit.attacks.ept.classification.XGBClassifier")
@patch("midst_toolkit.attacks.ept.classification.CatBoostClassifier")
def test_train_attack_classifier_tree_models(mock_catboost, mock_xgboost, classifier_type, attack_data):
    # Tests train_attack_classifier for XGBoost and CatBoost
    x_train, y_train, x_test, y_test = attack_data
    column_types = [ColumnType.ERROR]

    mock_model = MagicMock()
    mock_model.predict.return_value = np.zeros(10)
    mock_model.predict_proba.return_value = np.zeros((10, 2))
    if classifier_type == ClassifierType.XGBOOST:
        mock_xgboost.return_value = mock_model
    else:
        mock_catboost.return_value = mock_model

    results = train_attack_classifier(classifier_type, column_types, x_train, y_train, x_test, y_test)

    assert "prediction_results" in results
    assert "scores" in results
    assert "y_true" in results["prediction_results"]
    mock_model.fit.assert_called_once()
    mock_model.predict.assert_called_once()
    mock_model.predict_proba.assert_called_once()


@pytest.mark.parametrize("classifier_type", [ClassifierType.XGBOOST, ClassifierType.CATBOOST])
@patch("midst_toolkit.attacks.ept.classification.XGBClassifier")
@patch("midst_toolkit.attacks.ept.classification.CatBoostClassifier")
def test_train_attack_classifier_mismatched_data(mock_catboost, mock_xgboost, classifier_type, attack_data):
    # Tests that train_attack_classifier raises errors for mismatched data shapes
    x_train, y_train, x_test, y_test = attack_data
    column_types = [ColumnType.ERROR]

    mock_model = MagicMock()
    mock_model.predict.return_value = np.zeros(10)
    mock_model.predict_proba.return_value = np.zeros((10, 2))
    if classifier_type == ClassifierType.XGBOOST:
        mock_xgboost.return_value = mock_model
    else:
        mock_catboost.return_value = mock_model

    # Test mismatches
    with pytest.raises(AssertionError, match="Mismatch in number of training samples and labels"):
        train_attack_classifier(classifier_type, column_types, x_train.head(10), y_train, x_test, y_test)

    with pytest.raises(AssertionError, match="Mismatch in number of test samples and labels"):
        train_attack_classifier(classifier_type, column_types, x_train, y_train, x_test.head(5), y_test)

    x_test_wrong_features = x_test.rename(columns={"feature_error": "another_feature"})
    with pytest.raises(AssertionError, match="Mismatch in number of features between train and test sets"):
        train_attack_classifier(classifier_type, column_types, x_train, y_train, x_test_wrong_features, y_test)


@pytest.fixture
def input_dim():
    return 20


@pytest.fixture
def hidden_dim():
    return 10


@pytest.fixture
def sample_size():
    return 50


@pytest.fixture
def model(input_dim, hidden_dim):
    """Fixture to create a fresh model instance for each test."""
    return MLPClassifier(input_size=input_dim, hidden_size=hidden_dim, output_size=1, epochs=5, device=DEVICE)


@pytest.fixture
def dummy_data(sample_size, input_dim):
    """Fixture to create synthetic training data."""
    # Random features
    x = np.random.randn(sample_size, input_dim).astype(np.float32)
    # Random binary labels (0 or 1)
    y = np.random.randint(0, 2, size=(sample_size,)).astype(np.float32)
    return x, y


def test_initialization(model, input_dim, hidden_dim):
    """Test if the model initializes with correct layer dimensions."""
    assert isinstance(model.layers[0], nn.Linear)
    assert model.layers[0].in_features == input_dim
    assert model.layers[0].out_features == hidden_dim
    assert isinstance(model.layers[2], nn.Linear)
    assert model.layers[2].out_features == 1


def test_fit_updates_weights(model, dummy_data):
    """
    Test if calling fit() actually changes the model parameters.
    If weights don't change, training is broken.
    """
    x, y = dummy_data

    # Save a copy of the initial weights (specifically the first layer)
    initial_weights = model.layers[0].weight.data.clone()

    model.fit(x, y)

    # Check if weights have changed
    current_weights = model.layers[0].weight.data
    assert not torch.equal(initial_weights, current_weights), "Weights did not update after training"


def test_predict_proba_shape_and_values(model, dummy_data):
    """
    Test predict_proba output shape and probability constraints.
    Should return (n_samples, 2) and values between 0 and 1.
    """
    x, _ = dummy_data

    # We do not need to fit to test the shape of the output
    probas = model.predict_proba(x)

    # Check Shape: (n_samples, 2)
    assert probas.shape == (x.shape[0], 2)

    # Check Values: All between 0 and 1
    assert (probas >= 0).all() and (probas <= 1).all()

    # Check Sum: Probabilities across classes should sum to 1
    # We use np.isclose to handle floating point arithmetic
    row_sums = np.sum(probas, axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"


def test_predict_output_structure(model, dummy_data):
    """
    Test predict output structure.
    Should return (n_samples,) binary array.
    """
    x, _ = dummy_data

    predictions = model.predict(x)

    # Check Shape
    assert predictions.shape == (x.shape[0],)

    # Check content is binary (0.0 or 1.0)
    unique_vals = np.unique(predictions)
    assert np.isin(unique_vals, [0.0, 1.0]).all()


def test_train_attack_classifier_unsupported(attack_data):
    # Tests that an unsupported classifier type raises an assertion error
    x_train, y_train, x_test, y_test = attack_data
    column_types = [ColumnType.ERROR]

    with pytest.raises(ValueError):
        train_attack_classifier(MockClassifierType.UNSUPPORTED, column_types, x_train, y_train, x_test, y_test)
