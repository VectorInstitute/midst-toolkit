"""
This data collection module is tailored to the structure of the provided folders in
MIDST competition.
"""

from pathlib import Path

import pandas as pd

from midst_toolkit.attacks.black_box_single_table.ensemble_mia.config import (
    MIDST_DATA_PATH,
    POPULATION_PATH,
    dev_ids,
    final_ids,
    train_ids,
)
from midst_toolkit.attacks.black_box_single_table.ensemble_mia.data_processing.data_utils import (
    collect_challenge_points,
    collect_train_data,
    save_dataframe,
)


#  TODO: Maybe unify the following two functions.
def collect_attack_train_data(attack_types: list[tuple[str, bool]]) -> pd.DataFrame:
    """
    Collect train and challenge data of the specified attack type from
    from the provided data folders in the MIDST competition.

    Args:
        attack_types (list[Tuple[str, bool]]): list of tuples with attack names
            and a boolean indicating if it is a single table attack.

    Returns:
        pd.DataFrame: Collected train data as a DataFrame.
    """
    population = []
    for attack_name, is_single_table in attack_types:
        df_real = collect_train_data(
            attack_type=attack_name,
            data_dir=MIDST_DATA_PATH,
            data_id=train_ids,
            is_single_table=is_single_table,
        )

        population.append(df_real)

    return pd.concat(population).drop_duplicates()


def collect_attack_challenge_points(
    attack_types: list[tuple[str, bool]],
    datasets: list[str],
    folder_ids: list[list[int]],
) -> pd.DataFrame:
    """
    Collect challenge points for the specified attack types and datasets.
    The challenge points are collected from the MIDST data folders.

    Args:
        attack_types (list[Tuple[str, bool]]): List of tuples with attack names
            and a boolean indicating if it is a single table attack.
        datasets (list[str]): list of dataset names (e.g., "train", "dev", "final").
        folder_ids (list[list[int]]): list of lists containing folder ids for each attack type.

    Returns:
        pd.DataFrame: Collected challenge points as a DataFrame.
    """
    challenge_points = []
    assert len(attack_types) == len(folder_ids), "The length of attack_types and folder_ids should be the same."
    for i, (attack_name, _) in enumerate(attack_types):
        for dataset in datasets:
            challenge_points.append(
                collect_challenge_points(
                    attack_type=attack_name,
                    data_dir=MIDST_DATA_PATH,
                    data_id=folder_ids[i],
                    dataset=dataset,
                )
            )

    return pd.concat(challenge_points).drop_duplicates()


def collect_population_data(save_dir: Path = POPULATION_PATH) -> pd.DataFrame:
    """
    Collect the population data from the MIDST competition. Returns real data population
    that consists of the train data of all the attacks (black box and white box), and
    challange points from train, dev and final of tabddpm_black_box attack.
    The population data is saved in the provided path, and returned as a dataframe.
    """
    # Collect train data of all the attacks (back box and white box)
    # The first element of the tuple is the attack name and the second is a boolean
    # indicating if it is a single table attack.
    attack_types = [
        ("tabddpm_black_box", True),
        ("tabsyn_black_box", True),
        ("tabddpm_white_box", True),
        ("tabsyn_white_box", True),
        ("clavaddpm_black_box", False),
        ("clavaddpm_white_box", False),
    ]

    df_population = collect_attack_train_data(attack_types)
    # Drop ids.
    df_population_no_id = df_population.drop(columns=["trans_id", "account_id"])
    # Save the population data
    save_dataframe(df_population, save_dir, "population_all.csv")
    save_dataframe(df_population_no_id, save_dir, "population_all_no_id.csv")

    # Collect all the challenge points from train, dev and final of tabddpm_black_box
    datasets = ["train", "dev", "final"]
    folder_ids = [train_ids, dev_ids, final_ids]
    attack_types = [("tabddpm_black_box", True)]
    df_challenge = collect_attack_challenge_points(attack_types, datasets, folder_ids)
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
