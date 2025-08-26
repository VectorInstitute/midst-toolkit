from logging import INFO
from pathlib import Path

import pandas as pd

from midst_toolkit.common.logger import log


def save_dataframe(df: pd.DataFrame, file_path: Path, file_name: str) -> None:
    """
    Save a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): Dataframe to be saved.
        file_path (str): Save path.
        file_name (str): Name of the file to save the DataFrame as.
    """
    assert Path.exists(file_path), f"Path {file_path} does not exist."
    df.to_csv(file_path / file_name, index=False)
    log(INFO, f"DataFrame saved to {file_path / file_name}")


def load_dataframe(file_path: Path, file_name: str) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV file.

    Args:
        file_path (str): Path where the file is stored.
        file_name (str): Name of the file to load the DataFrame from.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    full_path = file_path / file_name
    assert Path.exists(full_path), f"File {full_path} does not exist."
    df = pd.read_csv(full_path)
    log(INFO, f"DataFrame loaded from {full_path}")
    return df


def collect_midst_attack_data(
    attack_type: str, data_dir: Path, data_split: str, dataset: str, data_config: dict
) -> pd.DataFrame:
    """
    Collect the real data in a specific setting of the provided MIDST challenge resources.

    Args:
        attack_type (str): The attack setting.
        data_dir (Path): The path where the data is stored.
        data_split (str): Indicates if this is train, dev, or final data.
        dataset (str): The dataset to be collected. Either "train" or "challenge".
        data_config (dict): Configuration dictionary containing data paths and file names.

    Returns:
        pd.DataFrame: The specified dataset in this setting.
    """
    # `data_id` is the folder numbering of each training or challenge dataset.
    assert data_split in [
        "train",
        "dev",
        "final",
    ], "data_split should be one of 'train', 'dev', or 'final'."
    data_id = data_config["folder_ids"][data_split]

    # Get file name based on the kind of dataset to be collected (i.e. train vs challenge).
    generation_name = attack_type.split("_")[0]
    if dataset == "challenge":
        file_name = data_config["challenge_data_file_name"]
    else:  # dataset == "train"
        # Multi-table attacks have different file names.
        file_name = (
            data_config["multi_table_train_data_file_name"]
            if generation_name == "clavaddpm"
            else data_config["single_table_train_data_file_name"]
        )
    assert file_name.split(".")[-1] == "csv", "File name should end with .csv."

    df_real = pd.DataFrame()
    for i in data_id:
        data_dir_ith = data_dir / attack_type / data_split / f"{generation_name}_{i}" / file_name
        df_real_ith = pd.read_csv(data_dir_ith)
        df_real = df_real_ith if i == 1 else pd.concat([df_real, df_real_ith])

    return df_real.drop_duplicates()
