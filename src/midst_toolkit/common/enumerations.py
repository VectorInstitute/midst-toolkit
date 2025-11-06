from enum import Enum


class TaskType(Enum):
    BINARY_CLASSIFICATION = "binclass"
    MULTICLASS_CLASSIFICATION = "multiclass"
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


class DomainDataType(Enum):
    """Possible types of domain data."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


class InfoDataType(Enum):
    """Possible types of column information data."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"


class ComputerRepresentation(Enum):
    """Possible types of computer representation for data values."""

    FLOAT = "Float"
