from enum import Enum


class TaskType(Enum):
    BINCLASS = "binclass"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"

    def __str__(self) -> str:
        """
        Return the string representation of the task type, which is the value of the enum.

        Returns:
            The string representation of the task type.
        """
        return self.value


class PredictionType(Enum):
    LOGITS = "logits"
    PROBS = "probs"


class DataSplit(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"
