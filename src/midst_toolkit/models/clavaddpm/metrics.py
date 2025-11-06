from typing import Any

import numpy as np
from scipy.special import expit, softmax
from sklearn.metrics import classification_report, r2_score, roc_auc_score, root_mean_squared_error

from midst_toolkit.common.enumerations import PredictionType, TaskType


def calculate_rmse(true_target: np.ndarray, predicted_target: np.ndarray, standard_deviation: float | None) -> float:
    """
    Calculate the root mean squared error (RMSE) of the predictions.

    Args:
        true_target: The true labels as a numpy array.
        predicted_target: The predicted labels as a numpy array.
        standard_deviation: The standard deviation of the labels. If provided, the RMSE is scaled by the standard
            deviation. This is typically done if the original targets were scaled down by the standard deviation during
            fitting/prediction. If None, the RMSE is calculated without this scaling.

    Returns:
        The RMSE of the predictions.
    """
    rmse = root_mean_squared_error(true_target, predicted_target)
    if standard_deviation is not None:
        return rmse * standard_deviation
    return rmse


def get_predicted_labels_and_probs(
    predicted_target: np.ndarray,
    task_type: TaskType,
    prediction_type: PredictionType | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Get the labels and probabilities from the predictions. If ``prediction_type`` is None, will return the predicted
    labels as is and the probabilities as None.

    Args:
        predicted_target: The predicted labels as a numpy array.
        task_type: The type of the task. Can be ``TaskType.BINARY_CLASSIFICATION`` or
            ``TaskType.MULTICLASS_CLASSIFICATION`` or None. Other task types are not supported.
        prediction_type: The type of the predictions. Currently supported types are ``PredictionType.LOGITS`` and
            ``PredictionType.PROBS``. If ``PredictionType.LOGITS`` then either they will be converted to binary
            probabilities with a sigmoid for ``TaskType.BINARY_CLASSIFICATION`` or multi-class
            probabilities with a softmax for ``TaskType.MULTICLASS_CLASSIFICATION``. ``PredictionType.PROBS``
            implies the predictions are already in probability form. If None, will return the predictions as
            labels and probabilities as None.

    Returns:
        A tuple with the labels and probabilities. The probabilities are None if the ``prediction_type`` is None.
    """
    assert task_type in {
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
    }, f"Unsupported task type: {task_type.value}"

    if prediction_type is None:
        return predicted_target, None

    if prediction_type == PredictionType.LOGITS:
        # expit applies a sigmoid
        prediction_probabilities = (
            expit(predicted_target)
            if task_type == TaskType.BINARY_CLASSIFICATION
            else softmax(predicted_target, axis=1)
        )
    elif prediction_type == PredictionType.PROBS:
        prediction_probabilities = predicted_target
    else:
        raise ValueError(f"Unsupported prediction_type: {prediction_type.value}")

    assert prediction_probabilities is not None
    assert prediction_probabilities.ndim == 1 or prediction_probabilities.shape[1] == 1
    predicted_labels = (
        np.round(prediction_probabilities)
        if task_type == TaskType.BINARY_CLASSIFICATION
        else prediction_probabilities.argmax(axis=1)
    )
    return predicted_labels.astype("int64"), prediction_probabilities


def calculate_metrics(
    true_target: np.ndarray,
    predicted_target: np.ndarray,
    task_type: TaskType,
    prediction_type: PredictionType | None,
    target_info: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate the metrics of the predictions.

    Example Usage: calculate_metrics(y_true, y_pred, TaskType.BINARY_CLASSIFICATION, PredictionType.LOGITS, {})

    Args:
        true_target: The true labels as a numpy array.
        predicted_target: The predicted labels as a numpy array.
        task_type: The type of the task.
        prediction_type: The type of the predictions.
        target_info: A dictionary with metadata about the labels.

    Returns:
        The metrics of the predictions as a dictionary with the following keys:
            If the task type is TaskType.REGRESSION:
                {
                    "rmse": The root mean squared error.
                    "r2": The R^2 score.
                }

            If the task type is TaskType.MULTICLASS_CLASSIFICATION, it will have a key for each label
            with the following metrics (result of sklearn.metrics.classification_report):
                {
                    "label-1": {
                        "precision": The precision of the label.
                        "recall": The recall of the label.
                        "f1-score": The F1 score of the label.
                        "support": The number of occurrences of this label in y_true.
                    },
                    "label-2": {...}
                    ...
                }

            If the task type is TaskType.BINARY_CLASSIFICATION, it will have a key for each label
            with the following metrics ((result of sklearn.metrics.classification_report),
            and an additional ROC AUC metric:
                {
                    "label-1": {
                        "precision": The precision of the label.
                        "recall": The recall of the label.
                        "f1-score": The F1 score of the label.
                        "support": The number of occurrences of this label in y_true.
                    },
                    "label-2": {...}
                    ...
                    "roc_auc": The ROC AUC score.
                }
    """
    if task_type == TaskType.REGRESSION:
        assert prediction_type is None
        assert "std" in target_info
        rmse = calculate_rmse(true_target, predicted_target, target_info["std"])
        r2 = r2_score(true_target, predicted_target)
        return {"rmse": rmse, "r2": r2}

    labels, probs = get_predicted_labels_and_probs(predicted_target, task_type, prediction_type)
    result = classification_report(true_target, labels, output_dict=True)
    assert isinstance(result, dict)

    if task_type == TaskType.BINARY_CLASSIFICATION:
        assert probs is not None, "Probabilities need to be defined to compute roc_acu"
        result["roc_auc"] = roc_auc_score(true_target, probs)

    return result
