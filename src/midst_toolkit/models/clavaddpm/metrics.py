import numpy as np
from scipy.special import expit, softmax

from midst_toolkit.common.enumerations import PredictionType, TaskType


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
