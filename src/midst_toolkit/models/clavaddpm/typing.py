from collections.abc import Callable
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
