from typing import Any


# TODO: Temporary, will switch to classes later
Configs = dict[str, Any]
Tables = dict[str, dict[str, Any]]
RelationOrder = list[tuple[str, str]]
GroupLengthsProbDicts = dict[tuple[str, str], dict[int, dict[int, float]]]
