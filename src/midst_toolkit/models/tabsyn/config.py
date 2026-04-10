import tomllib
from pathlib import Path
from typing import Any, Callable, cast


RawConfig = dict[str, Any]


CONFIG_NONE = "__none__"


def _replace(data: Any, condition: Callable[[Any], bool], value: Any) -> Any:
    """Replace the data with the value if the condition is met.

    Args:
        data: The data to replace. Can be a dictionary, a list or a scalar.
            If it is a dictionary or a list, the function will replace the values recursively.
        condition: The condition to check if the data should be replaced.
            The parameter is the data to check and the return value is a boolean.
        value: The value to replace the data with.

    Returns:
        The data with the values recursively replaced if the condition is met.
    """
    if isinstance(data, dict):
        return {k: _replace(v, condition, value) for k, v in data.items()}
    if isinstance(data, list):
        return [_replace(y, condition, value) for y in data]
    return value if condition(data) else data


def unpack_config(config: RawConfig) -> RawConfig:
    """Unpack the config by replacing the values with None if they are equal to CONFIG_NONE.

    Args:
        config: The config to unpack.

    Returns:
        The unpacked config.
    """
    config = cast(RawConfig, _replace(config, lambda x: x == CONFIG_NONE, None))
    return config


def load_config(path: Path | str) -> Any:
    """Load the config from the file.

    Will replace the values with None if they are equal to CONFIG_NONE.

    Args:
        path: The path to the config file.

    Returns:
        The config with the values replaced.
    """
    with open(path, "rb") as f:
        return unpack_config(tomllib.load(f))
