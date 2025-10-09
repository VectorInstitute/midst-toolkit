import json
import os
from logging import INFO, WARNING
from pathlib import Path
from typing import Any

import pandas as pd

from midst_toolkit.common.logger import log
from midst_toolkit.models.clavaddpm.data_loaders import (
    process_pipeline_data,
)


def save_dataframe(df: pd.DataFrame, file_path: Path, file_name: str) -> None:
    """
    Save a DataFrame to a CSV file.

    Args:
        df: DataFrame to be saved.
        file_path: Path where the file will be saved.
        file_name: Name of the file to save the DataFrame as.

    Returns:
        None
    """
    assert Path.exists(file_path), f"Path {file_path} does not exist."
    if Path.exists(file_path / file_name):
        log(
            WARNING,
            f"File {file_path / file_name} already exists and will be overwritten.",
        )
    df.to_csv(file_path / file_name, index=False)
    log(INFO, f"DataFrame saved to {file_path / file_name}")


def load_dataframe(file_path: Path, file_name: str) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV file.

    Args:
        file_path: Path where the file is stored.
        file_name: Name of the file to load the DataFrame from.

    Returns:
        Loaded dataframe.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    full_path = Path(file_path / file_name)
    if not Path.exists(full_path):
        raise FileNotFoundError(f"File {full_path} does not exist.")
    # Assert that the file is a CSV file
    assert full_path.suffix == ".csv", f"File {file_name} is not a CSV file."
    df = pd.read_csv(full_path)
    log(INFO, f"DataFrame loaded from {full_path}")
    return df


# TODO: Merge with the existing function in ``data_loaders.py`` .
# The following function is the slightly modified version of
# ``midst_toolkit.models.clavaddpm.data_loaders`` by the CITADEL & UQAM team.
def load_multi_table(
    data_dir: Path,
    train_df: pd.DataFrame | None = None,
    training_data_ratio: float = 1,
    verbose: bool = True,
) -> tuple[dict[str, Any], list[tuple[str, str]], dict[str, Any]]:
    """
    Load the multi-table dataset from the data directory.
    If a train_df is provided, it will be used as the training data.

    Args:
        data_dir: The directory to load the dataset from.
        train_df: Optional DataFrame to be used as the training data.
            If None, the function will look for a train.csv or ``f{table_name}.csv``
            file in the ``data_dir``.
        training_data_ratio: The ratio of the data to be used for training. Should be between 0 and 1.
            If it's equal to 1, it will only return the training set. Optional, default is 1.
        verbose: Whether to print verbose output. Optional, default is True.

    Returns:
        A tuple with 3 values:
            - The tables dictionary.
            - The relation order between the tables.
            - The dataset metadata dictionary.
    """
    with open(data_dir / "dataset_meta.json", "r") as f:
        dataset_meta = json.load(f)

    relation_order = dataset_meta["relation_order"]

    tables = {}

    for table, meta in dataset_meta["tables"].items():
        if train_df is None:
            if os.path.exists(os.path.join(data_dir, "train.csv")):
                train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
            else:
                train_df = pd.read_csv(os.path.join(data_dir, f"{table}.csv"))

        with open(data_dir / f"{table}_domain.json", "r") as f:
            domain = json.load(f)

        tables[table] = {
            "df": train_df,
            "domain": domain,
            "children": meta["children"],
            "parents": meta["parents"],
        }
        tables[table]["original_cols"] = list(tables[table]["df"].columns)
        tables[table]["original_df"] = tables[table]["df"].copy()
        id_cols = [col for col in tables[table]["df"].columns if "_id" in col]
        df_no_id = tables[table]["df"].drop(columns=id_cols)
        table_domain = tables[table]["domain"]

        _, info = process_pipeline_data(df_no_id, table_domain, training_data_ratio, verbose)
        tables[table]["info"] = info

    return tables, relation_order, dataset_meta
