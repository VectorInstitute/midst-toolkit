from enum import Enum
from logging import INFO
from typing import Any

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, auc, roc_curve
from torch import nn, optim
from xgboost import XGBClassifier

from midst_toolkit.common.logger import log
from midst_toolkit.common.variables import DEVICE


class ClassifierType(Enum):
    XGBOOST = "XGBoost"
    CATBOOST = "CatBoost"
    MLP = "MLP"


class ColumnType(Enum):
    ACTUAL = "actual"
    ERROR = "error"
    ERROR_RATIO = "error_ratio"
    ACCURACY = "accuracy"
    PREDICTION = "prediction"


def should_keep_column(column_name: str, column_types: list[ColumnType]) -> bool:
    """
    Determines if a column should be kept based on its suffix.

    Args:
        column_name: The name of the column.
        column_types: A list of `ColumnType` enums to check for.

    Returns:
        True if the column should be kept, False otherwise.
    """
    non_actual_suffixes = [s.value for s in ColumnType if s != ColumnType.ACTUAL]
    non_actual_suffixes_in_list = [s.value for s in column_types if s != ColumnType.ACTUAL]

    if column_name.endswith(tuple(non_actual_suffixes_in_list)):
        return True

    # 'actual' means we keep columns that don't have any of the other suffixes.
    if ColumnType.ACTUAL in column_types:
        return not column_name.endswith(tuple(non_actual_suffixes))

    return False


def filter_data(features_df: pd.DataFrame, column_types: list[str]) -> np.ndarray:
    """
    Filters columns from a single DataFrame based on specified suffixes.

    This function processes a pandas DataFrame, selecting columns based on
    suffixes that correspond to the types specified in `columns_list` (e.g.,
    'actual', 'error'). It then returns the data from these selected columns
    as a NumPy array.

    Args:
        features_df: The pandas DataFrame to process.
        column_types: A list of strings specifying the types of columns
            to select. (actual, error, error_ratio, accuracy, prediction)

    Returns:
        A NumPy array containing the data from the selected columns.
    """
    try:
        column_enums = [ColumnType(c) for c in column_types]
    except ValueError as e:
        raise ValueError(f"Invalid column type in `columns_list`. {e}") from e

    selected_columns = [column for column in features_df.columns if should_keep_column(column, column_enums)]

    return features_df[selected_columns].values


class MLPClassifier(nn.Module):
    def __init__(self, input_size: int = 100, hidden_size: int = 64, output_size: int = 1):
        """
        Creates the Multi-layer perceptron classifier. Defines a simple feedforward neural network with
        customizable input, hidden, and output sizes.

        Args:
            input_size: The number of features in the input data. Defaults to 100.
            hidden_size: The number of neurons in the hidden layer. Defaults to 64.
            output_size: The number of output neurons, typically 1 for binary classification. Defaults to 1.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the MLP.

        Args:
            x: The input tensor.

        Returns:
            The output tensor after passing through the network.
        """
        return self.layers(x).squeeze(dim=-1)


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray | None = None,
    device: torch.device = DEVICE,
    epochs: int = 10,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Trains a simple MLP classifier and optionally evaluates it on a test set.

    Args:
        x_train: Training data features.
        y_train: Training data labels.
        x_test: Test data features.
        device: The device to train the model on (e.g., 'cpu' or 'cuda').
        eval: If True, evaluates the model on the test set.
        epochs: Number of training epochs. Default is 10.

    Returns:
        A tuple containing:
        - The predicted labels for the test set (or None if eval is False).
        - The prediction probabilities for the test set (or None if eval is False).
    """
    input_size = x_train.shape[1]
    model = MLPClassifier(input_size=input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    x_train_tensor, y_train_tensor = (
        torch.tensor(x_train, dtype=torch.float32).to(device),
        torch.tensor(y_train, dtype=torch.float32).to(device),
    )

    # Train the model
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    y_pred, y_proba = None, None

    if x_test is not None:
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            # Get probabilities
            y_proba = model(x_test_tensor).cpu().numpy()
            # Convert probabilities to binary predictions
            y_pred = (y_proba > 0.5).astype(float)

    return y_pred, y_proba


def get_scores(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    fpr_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """
    Calculates and returns a dictionary of evaluation scores for a binary classifier.

    This function computes the accuracy, the Area Under the Receiver Operating Characteristic
    Curve (AUC-ROC), and the True Positive Rate (TPR) at specified False Positive Rate (FPR)
    thresholds.

    Args:
        y_true: Ground truth binary labels.
        y_proba: Predicted probabilities for the positive class.
        y_pred: Predicted binary labels.
        fpr_thresholds: A list of FPR values at which to calculate the TPR.
                        Defaults to [0.1, 0.01, 0.001].

    Returns:
        A dictionary containing the calculated scores: 'accuracy', 'AUC-ROC', and
        'TPR at FPR {threshold}' for each specified threshold.
    """
    if fpr_thresholds is None:
        fpr_thresholds = [0.1, 0.01, 0.001]

    accuracy = accuracy_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_roc = auc(fpr, tpr)

    scores = {"accuracy": accuracy, "AUC-ROC": auc_roc}

    # Compute TPR at specific FPR thresholds
    for threshold in fpr_thresholds:
        # Find the highest TPR for FPRs less than the threshold
        valid_tpr = tpr[fpr < threshold]
        tpr_at_fpr = valid_tpr.max() if valid_tpr.size > 0 else 0.0
        scores[f"TPR at FPR {threshold * 100}"] = tpr_at_fpr

    return scores


def train_attack_classifier(
    classifier_type: ClassifierType,
    column_types: list[str],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, dict]:
    """
    Trains a specified classifier for a membership inference attack.
    This function takes training and testing data, selects the subset of features given in
    the provided column list, and trains a classifier (XGBoost, CatBoost, or MLP) to
    distinguish between member and non-member data points. It then evaluates the
    classifier on the test set and returns the prediction results and performance scores.

    Args:
        classifier_type: The type of classifier to train.
            Supported values are "XGBoost", "CatBoost", and "MLP".
        column_types: A list of column type (actual, error, error_ratio, accuracy, prediction)
            to be used as features for training the classifier.
        x_train: The feature data for the training set.
        y_train: The labels for the training set (membership status).
        x_test: The feature data for the test set.
        y_test: The labels for the test set (membership status).

    Returns:
        A dictionary containing the results. It has two keys:
            - "prediction_results": A dictionary with the true labels ('y_true'),
              predicted probabilities ('y_proba'), and predicted labels ('y_pred').
            - "scores": A dictionary of performance metrics, including accuracy,
              AUC, and TPR at various FPR thresholds.
    """
    log(INFO, f"Training {classifier_type.value} classifier using features from column types: {column_types}")

    all_results: dict[str, Any] = {}

    x_train_processed = filter_data(x_train, column_types)
    y_train_processed = y_train.to_numpy()

    x_test_processed = filter_data(x_test, column_types)
    y_test_processed = y_test.to_numpy()

    assert x_train_processed.shape[0] == y_train_processed.shape[0], (
        "Mismatch in number of training samples and labels"
    )
    assert x_test_processed.shape[0] == y_test_processed.shape[0], "Mismatch in number of test samples and labels"
    assert x_train_processed.shape[1] == x_test_processed.shape[1], (
        "Mismatch in number of features between train and test sets"
    )

    y_pred, y_proba = None, None

    if classifier_type == ClassifierType.XGBOOST:
        model = XGBClassifier()
        model.fit(x_train_processed, y_train_processed)
        y_pred = model.predict(x_test_processed)
        y_proba = model.predict_proba(x_test_processed)[:, 1]

    elif classifier_type == ClassifierType.CATBOOST:
        model = CatBoostClassifier(verbose=0)
        model.fit(x_train_processed, y_train_processed)
        y_pred = model.predict(x_test_processed)
        y_proba = model.predict_proba(x_test_processed)[:, 1]

    elif classifier_type == ClassifierType.MLP:
        y_pred, y_proba = train_mlp(x_train_processed, y_train_processed, x_test_processed, DEVICE)

    assert y_pred is not None and y_proba is not None, (
        "Predictions and probabilities should not be None to get scores."
    )

    prediction_results = {
        "y_true": y_test_processed,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }

    fpr_thresholds = [0.1, 0.01, 0.001]

    all_results["prediction_results"] = prediction_results
    all_results["scores"] = get_scores(y_test_processed, y_proba, y_pred, fpr_thresholds)

    return all_results
