import json
from pathlib import Path
from typing import Any

import pandas as pd
from sdv.metadata import SingleTableMetadata  # type: ignore[import-untyped]


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


def get_single_table_svd_metadata(
    data: pd.DataFrame,
    domain_dictionary: dict[str, Any] | None = None,
) -> tuple[SingleTableMetadata, pd.DataFrame]:
    """
    Get the metadata for a single-table dataset for SDV models.

    Args:
        data: The dataframe containing the data.
        domain_dictionary: The domain dictionary containing metadata about the data columns.

    Returns:
        A tuple containing the metadata and the dataframe without the id columns.
    """
    metadata = SingleTableMetadata()
    data_without_ids = data.drop(columns=[column_name for column_name in data.columns if "_id" in column_name])
    metadata.detect_from_dataframe(data_without_ids)  # Starts up the metadata info from the dataframe's columns.

    if domain_dictionary is not None:
        for column_name in data_without_ids.columns:
            if domain_dictionary[column_name]["type"] == "discrete":
                if domain_dictionary[column_name]["size"] < 1000:
                    metadata.update_column(
                        column_name=column_name,
                        sdtype="categorical",
                    )
                else:
                    metadata.update_column(
                        column_name=column_name,
                        sdtype="numerical",
                    )
            else:
                metadata.update_column(
                    column_name=column_name,
                    sdtype="numerical",
                )

    metadata.remove_primary_key()

    return metadata, data_without_ids
