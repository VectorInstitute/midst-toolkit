"""
This data collection script is tailored to the structure of the provided folders in
MIDST competition.
"""

from enum import Enum
from logging import INFO
from pathlib import Path
from typing import Literal

import pandas as pd
from omegaconf import DictConfig

from midst_toolkit.attacks.ensemble.data_utils import load_dataframe, save_dataframe
from midst_toolkit.common.logger import log


class AttackType(Enum):
    """Enum for the different attack types."""

    TABDDPM_BLACK_BOX = "tabddpm_black_box"
    TABDDPM_WHITE_BOX = "tabddpm_white_box"
    TABSYN_BLACK_BOX = "tabsyn_black_box"
    TABSYN_WHITE_BOX = "tabsyn_white_box"
    CLAVADDPM_BLACK_BOX = "clavaddpm_black_box"
    CLAVADDPM_WHITE_BOX = "clavaddpm_white_box"
    # Experiment attack types based on experiment settings
    TABDDPM_5K = "tabddpm_trained_with_5k"
    TABDDPM_10K = "tabddpm_trained_with_10k"
    TABDDPM_20K = "tabddpm_trained_with_20k"
    TABDDPM_50K = "tabddpm_trained_with_50k"
    TABDDPM_100K = "tabddpm_trained_with_100k"


DatasetType = Literal["train", "challenge"]


def expand_ranges(ranges: list[tuple[int, int]]) -> list[int]:
    """
    Reads a list of tuples representing ranges and expands them into a flat list of integers.

    Args:
        ranges: List of tuples, where each tuple contains two integers (start, end).

    Returns:
        A flat list of integers covering the ranges.
    """
    expanded: list = []
    for list_range in ranges:
        start, end = list_range
        expanded.extend(range(start, end))
    return expanded


def collect_data_from_path_range(data_path: Path, data_range: list[tuple[int, int]], generation_name: str, file_name: str) -> pd.DataFrame:
    collected_data = pd.DataFrame()
    data_id = expand_ranges(data_range)
    for i in data_id:
        data_path_ith = data_path / f"{generation_name}_{i}"
        # Will raise FileNotFoundError if the file does not exist or if it is not a CSV file.
        collected_data_ith = load_dataframe(data_path_ith, file_name)
        collected_data = collected_data_ith if collected_data.empty else pd.concat([collected_data, collected_data_ith])
    return collected_data.drop_duplicates()


def collect_midst_attack_data(
    attack_type: AttackType,
    data_dir: Path,
    split_folder: str,
    dataset: DatasetType,
    data_processing_config: DictConfig,
) -> pd.DataFrame:
    """
    Collect the real data in a specific setting of the provided MIDST challenge resources.

    Args:
        attack_type: The attack setting.
        data_dir: The path where the data is stored.
        split_folder: Indicates the folder name to collect data from for a specific data split.
            ``split_folder`` should exist under ``data_dir / attack_type.value`` and
            f"{generation_name}_{i}" should be located under ``split_folder``.
        dataset: The dataset to be collected. Either "train" or "challenge".
        data_processing_config: Configuration dictionary containing data specific information.

    Returns:
        pd.DataFrame: The specified dataset in this setting.
    """
    assert dataset in {
        "train",
        "challenge",
    }, "Only 'train' and 'challenge' collection is supported."
    # `data_id` is the folder numbering of each training or challenge dataset,
    #  and is defined with the provided config.
    data_range = data_processing_config.folder_ranges[split_folder]

    # Get file name based on the kind of dataset to be collected (i.e. train vs challenge).
    # TODO: Make the below parsing a bit more robust and less brittle
    generation_name = attack_type.value.split("_")[0]
    if dataset == "challenge":
        file_name = data_processing_config.challenge_data_file_name
    else:
        # Multi-table attacks have different file names.
        file_name = (
            data_processing_config.multi_table_train_data_file_name
            if generation_name == "clavaddpm"
            else data_processing_config.single_table_train_data_file_name
        )
    data_path = data_dir / attack_type.value / split_folder
    return collect_data_from_path_range(data_path, data_range, generation_name, file_name)


# TODO: find a better name for dataset argument in the functions below.
def collect_midst_data(
    midst_data_input_dir: Path,
    attack_types: list[AttackType],
    split_folders: list[str],
    dataset: DatasetType,
    data_processing_config: DictConfig,
) -> pd.DataFrame:
    """
    Collect train or challenge data of the specified attack type from the provided data folders
    in the MIDST competition. The data is going to be collected from all the folders specified
    in ``split_folders`` argument under each attack type folder. For example, if ``split_folders``
    contains `train` and `dev`, the function collects data from both `train` and `dev` folders
    under each attack type folder. For more information about the data collection structure, see
    the implementation of ``collect_midst_attack_data`` function.

    Args:
        midst_data_input_dir: The path where the MIDST data folders are stored.
        attack_types: List of attack types for data collection.
        split_folders: A list indicating the folder names to collect data splits from. These folders should exist
            under each attack type folder where we collect model's data from. For example, it could
            contain strings like `train`, `dev`, `final`, or `test` based on the directory structure.
        dataset: The dataset to be collected. Either `train` or `challenge`.
        data_processing_config: Configuration dictionary containing data paths and file names.

    Returns:
        Collected train or challenge data as a dataframe.
    """
    assert dataset in {"train", "challenge"}, "Only 'train' and 'challenge' collection is supported."
    population = []
    for attack_type in attack_types:
        for split_folder in split_folders:
            df_real = collect_midst_attack_data(
                attack_type=attack_type,
                data_dir=midst_data_input_dir,
                split_folder=split_folder,
                dataset=dataset,
                data_processing_config=data_processing_config,
            )

            population.append(df_real)

    return pd.concat(population).drop_duplicates()


def collect_population_data_ensemble(
    midst_data_input_dir: Path,
    data_processing_config: DictConfig,
    save_dir: Path,
    base_population: pd.DataFrame | None = None,
    population_splits: list[str] | None = None,
    challenge_splits: list[str] | None = None,
) -> pd.DataFrame:
    """
    Collect the population data from the MIDST competition based on Ensemble Attack implementation.
    Returns real data population that consists of the train data of all the attacks
    (black box and white box) as specified in ``data_processing_config.population_attack_data_types_to_collect``
    , and challenge points from `train`, `dev` and `final` of attacks as specified by
    ``data_processing_config.challenge_attack_data_types_to_collect``. If ``base_population`` is not None,
    the collected population data will be concatenated with ``base_population`` to be large enough for
    the attack (especially DOMIAS), then is saved in the provided path, and returned as a dataframe.

    Args:
        midst_data_input_dir: The path where the MIDST data folders are stored.
        data_processing_config: Configuration dictionary containing data information and file names.
        save_dir: The path where the collected population data should be saved.
        base_population: Path to a large dataset to be concatenated with the collected population data
            in this function. In experiments, the original attack's population data (800k records) collected by
            the attacker team is used as the base population. This data is concatenated with the newly collected
            population data to form a larger population for the attack (especially needed for DOMIAS). If None,
            only the newly collected population data is used, which may not yield the expected attack performance.
        population_splits: A list containing the folder names under attack folders that are
            considered for population collection. If None, the default list of ``["train"]`` is set in the
            function based on the original attack implementation.
        challenge_splits:  list containing the folder names under attack folders that are
            considered for challenge data collection. If None, the default list of ``["train", "dev", "final"]``
            is set in the function based on the original attack implementation.

    Returns:
        The collected population data as a dataframe.
    """
    # Population data will be saved under ``save_dir``.
    save_dir.mkdir(parents=True, exist_ok=True)

    if population_splits is None:
        population_splits = ["train"]
    if challenge_splits is None:
        # Original Ensemble collects all the challenge points from train, dev and final of "tabddpm_black_box" attack.
        challenge_splits = ["train", "dev", "final"]

    # Ensemble Attack collects train data of all the attack types (black box and white box)
    population_attack_names = data_processing_config.population_attack_data_types_to_collect
    # Provided attack name are valid based on AttackType enum
    population_attack_types = [AttackType(attack_name) for attack_name in population_attack_names]

    df_population_experiment = collect_midst_data(
        midst_data_input_dir,
        population_attack_types,
        split_folders=population_splits,
        dataset="train",
        data_processing_config=data_processing_config,
    )

    log(INFO, f"Collected experiment population data length before concatenation: {len(df_population_experiment)}")

    if base_population is not None:
        df_population = pd.concat([df_population_experiment, base_population]).drop_duplicates()
        log(INFO, f"Concatenated population data length: {len(df_population)}")
    else:
        df_population = df_population_experiment
        log(
            INFO,
            "base_population is None, only the newly collected population data is used.",
        )

    # Drop ids.
    df_population_no_id = df_population.drop(columns=["trans_id", "account_id"])
    # Save the population data
    save_dataframe(df_population, save_dir, "population_all.csv")
    save_dataframe(df_population_no_id, save_dir, "population_all_no_id.csv")

    challenge_attack_names = data_processing_config.challenge_attack_data_types_to_collect
    challenge_attack_types = [AttackType(attack_name) for attack_name in challenge_attack_names]

    df_challenge = collect_midst_data(
        midst_data_input_dir,
        attack_types=challenge_attack_types,
        split_folders=challenge_splits,
        dataset="challenge",
        data_processing_config=data_processing_config,
    )
    log(INFO, f"Collected challenge data length: {len(df_challenge)} from splits: {challenge_splits}")
    # In some cases, the location of target models are totally different from train models, therefore 
    # to collect the test challenge points, we need to look into the attack folders directly.
    # This offers flexibility to the data folder structure.
    # This lets us load challenge points from any directory, even if it's not the same as train.
    if "test_challenge_data_path_for_training" in data_processing_config:
        test_challenge_data = collect_data_from_path_range(
            data_path=Path(data_processing_config.test_challenge_data_path_for_training),
            data_range=data_processing_config.folder_ranges["final"],
            generation_name="tabddpm",
            file_name=data_processing_config.challenge_data_file_name,
        )
        df_challenge = pd.concat([df_challenge, test_challenge_data]).drop_duplicates()
        log(INFO, f"Added challenge data of length: {len(test_challenge_data)} from target models directory.")
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
    return df_population_with_challenge
