import tomllib
from pathlib import Path
from typing import Any, cast


RawConfig = dict[str, Any]


CONFIG_NONE = "__none__"


def _replace(data, condition, value):
    def do(x):
        if isinstance(x, dict):
            return {k: do(v) for k, v in x.items()}
        if isinstance(x, list):
            return [do(y) for y in x]
        return value if condition(x) else x

    return do(data)


def unpack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x == CONFIG_NONE, None))
    return config


def load_config(path: Path | str) -> Any:
    with open(path, "rb") as f:
        return unpack_config(tomllib.load(f))
