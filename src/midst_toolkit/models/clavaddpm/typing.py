from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np
from torch import nn


# TODO: Temporary, will switch to classes later
Configs = dict[str, Any]
Tables = dict[str, dict[str, Any]]
RelationOrder = list[tuple[str, str]]
GroupLengthsProbDicts = dict[tuple[str, str], dict[int, dict[int, float]]]
ArrayDict = dict[str, np.ndarray]
ModuleType = str | Callable[..., nn.Module]


class ClusteringMethod(Enum):
    """Possioble clustering methods for multi-table training."""

    KMEANS = "kmeans"
    GMM = "gmm"
    KMEANS_AND_GMM = "kmeans_and_gmm"
    VARIATIONAL = "variational"
