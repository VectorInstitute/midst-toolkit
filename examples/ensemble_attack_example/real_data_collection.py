"""
This data collection script is tailored to the structure of the provided folders in
MIDST competition.
"""
from pathlib import Path
import pandas as pd
from omegaconf import DictConfig

from src.midst_toolkit.attacks.ensemble.utils import (
    save_dataframe,
)


def expand_ranges(ranges):
    expanded = []
    for r in ranges:
        start, end = r
        expanded.extend(range(start, end))
    return expanded


def collect_midst_attack_data(
    attack_type: str,
    data_dir: Path,
    data_split: str,
    dataset: str,
    data_processing_config: DictConfig,
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
    data_id = expand_ranges(data_processing_config.folder_ranges[data_split])

    # Get file name based on the kind of dataset to be collected (i.e. train vs challenge).
    generation_name = attack_type.split("_")[0]
    if dataset == "challenge":
        file_name = data_processing_config.challenge_data_file_name
    else:  # dataset == "train"
        # Multi-table attacks have different file names.
        file_name = (
            data_processing_config.multi_table_train_data_file_name
            if generation_name == "clavaddpm"
            else data_processing_config.single_table_train_data_file_name
        )
    assert file_name.split(".")[-1] == "csv", "File name should end with .csv."

    df_real = pd.DataFrame()
    for i in data_id:
        data_dir_ith = (
            data_dir / attack_type / data_split / f"{generation_name}_{i}" / file_name
        )
        df_real_ith = pd.read_csv(data_dir_ith)
        df_real = df_real_ith if i == 1 else pd.concat([df_real, df_real_ith])

    return df_real.drop_duplicates()


# TODO: find a better name for dataset argument in the functions below.
def collect_midst_data(
    midst_data_input_dir: Path,
    attack_types: list[str],
    data_splits: list[str],
    dataset: str,
    data_config: DictConfig,
) -> pd.DataFrame:
    """
    Collect train or challenge data of the specified attack type from the provided data folders
    in the MIDST competition.

    Args:
        attack_types (list[str]): List of attack names to be collected.
        data_splits (list[str]): A list indicating the data split to be collected.
            Could be any of train, dev, or final data splits.
        dataset (str): The dataset to be collected. Either "train" or "challenge".
        data_config (dict): Configuration dictionary containing data paths and file names.

    Returns:
        pd.DataFrame: Collected train or challenge data as a DataFrame.
    """
    assert dataset in [
        "train",
        "challenge",
    ], " Only 'train' and 'challenge' collection is supported."
    population = []
    for attack_name in attack_types:
        for data_split in data_splits:
            df_real = collect_midst_attack_data(
                attack_type=attack_name,
                data_dir=midst_data_input_dir,
                data_split=data_split,
                dataset=dataset,
                data_processing_config=data_config,
            )

        population.append(df_real)

    return pd.concat(population).drop_duplicates()


def collect_population_data_ensemble(
    midst_data_input_dir: Path,
    data_processing_config: DictConfig,
    save_dir: Path,
) -> pd.DataFrame:
    """
    Collect the population data from the MIDST competition based on ensemble mia implementation.
    Returns real data population that consists of the train data of all the attacks
    (black box and white box), and challenge points from train, dev and final of
    "tabddpm_black_box" attack. The population data is saved in the provided path,
    and returned as a dataframe.

    Args:
        data_config (dict): Configuration dictionary containing data paths and file names.
        attack_types (list[str] | None): List of attack names to be collected.
            If None, all the attacks are collected based on ensemble mia implementation.

    Returns:
        pd.DataFrame: The collected population data.
    """

    # Ensemble Attack collects train data of all the attack types (back box and white box)
    attack_types = data_processing_config.collect_attack_data_types
    df_population = collect_midst_data(
        midst_data_input_dir,
        attack_types,
        data_splits=["train"],
        dataset="train",
        data_config=data_processing_config,
    )
    # Drop ids.
    df_population_no_id = df_population.drop(columns=["trans_id", "account_id"])
    # Save the population data
    save_dataframe(df_population, save_dir, "population_all.csv")
    save_dataframe(df_population_no_id, save_dir, "population_all_no_id.csv")

    # Collect all the challenge points from train, dev and final of "tabddpm_black_box" attack.
    challenge_attack_types = ["tabddpm_black_box"]
    df_challenge = collect_midst_data(
        midst_data_input_dir,
        attack_types=challenge_attack_types,
        data_splits=["train", "dev", "final"],
        dataset="challenge",
        data_config=data_processing_config,
    )
    # Save the challenge points
    save_dataframe(df_challenge, save_dir, "challenge_points_all.csv")

    # Population data without the challenge points
    df_population_no_challenge = df_population[~df_population["trans_id"].isin(df_challenge["trans_id"])]
    save_dataframe(df_population_no_challenge, save_dir, "population_all_no_challenge.csv")
    # Remove ids
    df_population_no_challenge_no_id = df_population_no_challenge.drop(columns=["trans_id", "account_id"])
    save_dataframe(
        df_population_no_challenge_no_id,
        save_dir,
        "population_all_no_challenge_no_id.csv",
    )

    # Population data with all the challenge points
    df_population_with_challenge = pd.concat([df_population_no_challenge, df_challenge])
    save_dataframe(df_population_with_challenge, save_dir, "population_all_with_challenge.csv")
    # Remove ids
    df_population_with_challenge_no_id = df_population_with_challenge.drop(columns=["trans_id", "account_id"])
    save_dataframe(
        df_population_with_challenge_no_id,
        save_dir,
        "population_all_with_challenge_no_id.csv",
    )

    return df_population_with_challenge_no_id
