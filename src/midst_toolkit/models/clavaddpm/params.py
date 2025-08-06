from collections.abc import Callable
from typing import Any

import numpy as np
from torch import nn


# TODO: Temporary, will wtich to classes later
Configs = dict[str, Any]
Tables = dict[str, dict[str, Any]]
RelationOrder = list[tuple[str, str]]

ArrayDict = dict[str, np.ndarray]
ModuleType = str | Callable[..., nn.Module]
