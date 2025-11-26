import json
from pathlib import Path


def get_table_name(base_data_dir: Path) -> str:
    """
    Get the name of the table from the dataset metadata.

    Args:
        base_data_dir: The base directory containing the dataset metadata.

    Returns:
        The name of the table.
    """
    with open(Path(base_data_dir) / "dataset_meta.json", "r") as f:
        dataset_meta = json.load(f)

    assert len(dataset_meta["tables"]) == 1, (
        "Only one table is supported for single-table training. "
        f"Got {len(dataset_meta['tables'])} tables: {dataset_meta['tables'].keys()}"
    )

    return list(dataset_meta["tables"].keys())[0]
