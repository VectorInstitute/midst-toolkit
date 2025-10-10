from logging import INFO, WARNING
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
