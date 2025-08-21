import logging
from pathlib import Path

import pandas as pd


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
    logging.info(f"DataFrame saved to {file_path / file_name}")


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
    logging.info(f"DataFrame loaded from {full_path}")
    return df


# TODO: Maybe unify the following two functions.
def collect_train_data(attack_type: str, data_dir: Path, data_id: list[int], is_single_table: bool) -> pd.DataFrame:
    """
    Collect the real training data in a specific setting.

    Args:
        attack_type (str): The attack setting.
        data_dir (Path): The path where the data is stored.
        data_id (range): The folder numbering of each training dataset.
        is_single_table (bool): Whether it is a single table attack setting.

    Returns:
        pd.DataFrame: All the real training data in this setting.
    """
    gen_name = attack_type.split("_")[0]
    df_real = pd.DataFrame()
    for i in data_id:
        if is_single_table:
            data_dir_ith = data_dir / attack_type / "train" / f"{gen_name}_{i}" / "train_with_id.csv"
        else:
            data_dir_ith = data_dir / attack_type / "train" / f"{gen_name}_{i}" / "trans.csv"

        df_real_ith = pd.read_csv(data_dir_ith)
        df_real = df_real_ith if i == 1 else pd.concat([df_real, df_real_ith])

    return df_real.drop_duplicates()


def collect_challenge_points(attack_type: str, data_dir: Path, data_id: list[int], dataset: str) -> pd.DataFrame:
    """
    Collect the challenge points in a specific setting.

    Args:
        attack_type (str): The setting to attack.
        data_dir (Path): The path where the data is stored.
        data_id (range): The numbering of each challenge dataset.
        dataset (str): Indicates if this is train, dev, or final data.

    Returns:
        pd.DataFrame: All the challenge points in this setting.
    """
    gen_name = attack_type.split("_")[0]
    df_test = pd.DataFrame()

    for idx, i in enumerate(data_id):
        data_dir_ith = data_dir / attack_type / dataset / f"{gen_name}_{i}" / "challenge_with_id.csv"

        df_test_ith = pd.read_csv(data_dir_ith)

        df_test = df_test_ith if idx == 0 else pd.concat([df_test, df_test_ith])

    return df_test.drop_duplicates()
