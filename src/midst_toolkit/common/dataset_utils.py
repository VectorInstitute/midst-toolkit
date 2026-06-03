from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_pickle(path: Path | str, **kwargs: Any) -> Any:
    """
    Load a pickle file.

    Args:
        path: The path to the pickle file.
        **kwargs: Additional arguments to pass to the pickle.loads function.

    Returns:
        The loaded pickle file.
    """
    return pickle.loads(Path(path).read_bytes(), **kwargs)


def dump_pickle(obj: Any, path: Path | str, **kwargs: Any) -> None:
    """
    Dump an object into a pickle file.

    Args:
        obj: The object to dump.
        path: The path to the pickle file.
        **kwargs: Additional arguments to pass to the pickle.dumps function.
    """
    Path(path).write_bytes(pickle.dumps(obj, **kwargs))


def get_category_sizes(features: torch.Tensor | np.ndarray) -> list[int]:
    """
    Get the size of the categories in the features tensor or array provided by counting the number of
    unique values in each column.

    Args:
        features: The data from which to extract category sizes.

    Returns:
        A list with the category sizes in the data.
    """
    columns_list = features.T.cpu().tolist() if isinstance(features, torch.Tensor) else features.T.tolist()
    return [len(set(column)) for column in columns_list]
