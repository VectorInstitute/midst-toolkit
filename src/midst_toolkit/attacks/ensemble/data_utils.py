from logging import INFO
from pathlib import Path

import pandas as pd

from midst_toolkit.common.logger import log


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
    """
    full_path = file_path / file_name
    assert Path.exists(full_path), f"File {full_path} does not exist."
    df = pd.read_csv(full_path)
    log(INFO, f"DataFrame loaded from {full_path}")
    return df
