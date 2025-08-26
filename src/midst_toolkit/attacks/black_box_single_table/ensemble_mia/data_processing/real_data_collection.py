"""
This data collection module is tailored to the structure of the provided folders in
MIDST competition.
"""

import pandas as pd

from midst_toolkit.attacks.black_box_single_table.ensemble_mia.data_processing.data_utils import (
    collect_midst_attack_data,
    save_dataframe,
)


# TODO: find a better name for dataset argument in the functions below.
def collect_midst_data(
    attack_types: list[str], data_splits: list[str], dataset: str, data_config: dict
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
                data_dir=data_config["midst_data_path"],
                data_split=data_split,
                dataset=dataset,
                data_config=data_config,
            )

        population.append(df_real)

    return pd.concat(population).drop_duplicates()


def collect_population_data_ensemble_mia(
    data_config: dict,
    attack_types: list[str] | None = None,
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
    save_dir = data_config["population_path"]

    # Collect train data of all the attacks (back box and white box)
    if attack_types is None:
        attack_types = [
            "tabddpm_black_box",
            "tabsyn_black_box",
            "tabddpm_white_box",
            "tabsyn_white_box",
            "clavaddpm_black_box",
            "clavaddpm_white_box",
        ]

    df_population = collect_midst_data(attack_types, data_splits=["train"], dataset="train", data_config=data_config)
    # Drop ids.
    df_population_no_id = df_population.drop(columns=["trans_id", "account_id"])
    # Save the population data
    save_dataframe(df_population, save_dir, "population_all.csv")
    save_dataframe(df_population_no_id, save_dir, "population_all_no_id.csv")

    # Collect all the challenge points from train, dev and final of "tabddpm_black_box" attack.
    challenge_attack_types = ["tabddpm_black_box"]
    df_challenge = collect_midst_data(
        attack_types=challenge_attack_types,
        data_splits=["train", "dev", "final"],
        dataset="challenge",
        data_config=data_config,
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
