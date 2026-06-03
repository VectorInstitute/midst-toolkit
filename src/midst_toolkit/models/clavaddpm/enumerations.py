from enum import Enum


Relation = tuple[str | None, str]
RelationOrder = list[Relation]
GroupLengthProbDict = dict[int, dict[int, float]]
GroupLengthsProbDicts = dict[Relation, GroupLengthProbDict]


class ClusteringMethod(Enum):
    """Possible clustering methods for multi-table training."""

    KMEANS = "kmeans"
    GMM = "gmm"
    KMEANS_AND_GMM = "kmeans_and_gmm"
    VARIATIONAL = "variational"


class ReductionMethod(Enum):
    """Possible methods of reduction."""

    MEAN = "mean"
    SUM = "sum"
    NONE = "none"


class DataAndKeyNormalizationType(Enum):
    """Possible types of normalization for data and primary keys when clustering."""

    MINMAX = "minmax"
    QUANTILE = "quantile"


class ModuleType(Enum):
    """Possible types of modules for the MLP or ResNet models."""

    REGLU = "ReGLU"
    GEGLU = "GEGLU"
    RELU = "ReLU"
    BATCH_NORM_1D = "BatchNorm1d"
