from enum import Enum


class TaskType(Enum):
    BINCLASS = "binclass"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"

    def __str__(self) -> str:
        """Return the string value of the enum."""
        return self.value


class PredictionType(Enum):
    LOGITS = "logits"
    PROBS = "probs"
