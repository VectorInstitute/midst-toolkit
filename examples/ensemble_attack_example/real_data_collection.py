"""
This data collection script is tailored to the structure of the provided folders in
MIDST competition.
"""

from enum import Enum
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from midst_toolkit.attacks.ensemble.data_utils import load_dataframe, save_dataframe


class AttackType(Enum):
    """Enum for the different attack types."""

    TABDDPM_BLACK_BOX = "tabddpm_black_box"
    TABDDPM_WHITE_BOX = "tabddpm_white_box"
    TABSYN_BLACK_BOX = "tabsyn_black_box"
    TABSYN_WHITE_BOX = "tabsyn_white_box"
    CLAVADDPM_BLACK_BOX = "clavaddpm_black_box"
    CLAVADDPM_WHITE_BOX = "clavaddpm_white_box"


def expand_ranges(ranges: list[tuple[int, int]]) -> list[int]:
    """
    Reads a list of tuples representing ranges and expands them into a flat list of integers.

    Args:
        ranges: List of tuples, where each tuple contains two integers (start, end).

    Returns:
        A flat list of integers covering the ranges.
    """
    expanded: list[int] = []
    for r in ranges:
        start, end = r
        expanded.extend(range(start, end))
    return expanded


def collect_midst_attack_data(
    attack_type: AttackType,
    data_dir: Path,
    data_split: str,
    dataset: str,
    data_processing_config: DictConfig,
) -> pd.DataFrame:
    """
    Collect the real data in a specific setting of the provided MIDST challenge resources.

    Args:
        attack_type: The attack setting.
        data_dir: The path where the data is stored.
        data_split: Indicates if this is train, dev, or final data.
        dataset: The dataset to be collected. Either "train" or "challenge".
        data_processing_config: Configuration dictionary containing data specific information.

    Returns:
        pd.DataFrame: The specified dataset in this setting.
    """
    assert data_split in [
        "train",
        "dev",
        "final",
    ], "data_split should be one of 'train', 'dev', or 'final'."
    # `data_id` is the folder numbering of each training or challenge dataset,
    #  and is defined with the provided config.
    data_id = expand_ranges(data_processing_config.folder_ranges[data_split])

    # Get file name based on the kind of dataset to be collected (i.e. train vs challenge).
    # TODO: Make the below parsing a bit more robust and less brittle
    generation_name = attack_type.value.split("_")[0]
    if dataset == "challenge":
        file_name = data_processing_config.challenge_data_file_name
    else:  # dataset == "train"
        # Multi-table attacks have different file names.
        file_name = (
            data_processing_config.multi_table_train_data_file_name
            if generation_name == "clavaddpm"
            else data_processing_config.single_table_train_data_file_name
        )

    df_real = pd.DataFrame()
    for i in data_id:
        data_path_ith = data_dir / attack_type.value / data_split / f"{generation_name}_{i}"
        # Will raise FileNotFoundError if the file does not exist or if it is not a CSV file.
        df_real_ith = load_dataframe(data_path_ith, file_name)
        df_real = df_real_ith if i == 1 else pd.concat([df_real, df_real_ith])

    return df_real.drop_duplicates()


# TODO: find a better name for dataset argument in the functions below.
def collect_midst_data(
    midst_data_input_dir: Path,
    attack_types: list[AttackType],
    data_splits: list[str],
    dataset: str,
    data_processing_config: DictConfig,
) -> pd.DataFrame:
    """
    Collect train or challenge data of the specified attack type from the provided data folders
    in the MIDST competition.

    Args:
        midst_data_input_dir: The path where the MIDST data folders are stored.
        attack_types: List of attack types for data collection.
        data_splits: A list indicating the data split to be collected.
            Could be any of train, dev, or final data splits.
        dataset: The dataset to be collected. Either `train` or `challenge`.
        data_processing_config: Configuration dictionary containing data paths and file names.

    Returns:
        Collected train or challenge data as a dataframe.
    """
    assert dataset in {"train", "challenge"}, "Only 'train' and 'challenge' collection is supported."
    population = []
    for attack_type in attack_types:
        for data_split in data_splits:
            df_real = collect_midst_attack_data(
                attack_type=attack_type,
                data_dir=midst_data_input_dir,
                data_split=data_split,
                dataset=dataset,
                data_processing_config=data_processing_config,
            )

        population.append(df_real)

    return pd.concat(population).drop_duplicates()


def collect_population_data_ensemble(
    midst_data_input_dir: Path,
    data_processing_config: DictConfig,
    save_dir: Path,
) -> pd.DataFrame:
    """
    Collect the population data from the MIDST competition based on Ensemble Attack implementation.
    Returns real data population that consists of the train data of all the attacks
    (black box and white box), and challenge points from `train`, `dev` and `final` of
    "tabddpm_black_box" attack. The population data is saved in the provided path,
    and returned as a dataframe.

    Args:
        midst_data_input_dir: The path where the MIDST data folders are stored.
        data_processing_config: Configuration dictionary containing data information and file names.
        save_dir: The path where the collected population data should be saved.

    Returns:
        The collected population data as a dataframe.
    """
    # Ensemble Attack collects train data of all the attack types (black box and white box)
    attack_names = data_processing_config.collect_attack_data_types
    # Provided attack name are valid based on AttackType enum
    attack_types: list[AttackType] = [AttackType(attack_name) for attack_name in attack_names]

    df_population = collect_midst_data(
        midst_data_input_dir,
        attack_types,
        data_splits=["train"],
        dataset="train",
        data_processing_config=data_processing_config,
    )
    # Drop ids.
    df_population_no_id = df_population.drop(columns=["trans_id", "account_id"])
    # Save the population data
    save_dataframe(df_population, save_dir, "population_all.csv")
    save_dataframe(df_population_no_id, save_dir, "population_all_no_id.csv")

    # Collect all the challenge points from train, dev and final of "tabddpm_black_box" attack.
    challenge_attack_types = [AttackType.TABDDPM_BLACK_BOX]
    df_challenge = collect_midst_data(
        midst_data_input_dir,
        attack_types=challenge_attack_types,
        data_splits=["train", "dev", "final"],
        dataset="challenge",
        data_processing_config=data_processing_config,
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
