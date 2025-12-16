import os
from logging import INFO
from pathlib import Path
from typing import Any, cast

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from midst_toolkit.attacks.tartan_federer.tartan_federer_attack import tartan_federer_attack
from midst_toolkit.common.logger import log
from midst_toolkit.common.random import set_all_random_seeds, unset_all_random_seeds


def prepare_population_dataset_for_attack(
    model_indices: list[int],
    model_type: str,
    models_base_dir: Path,
    columns_for_deduplication: list[str],
) -> pd.DataFrame:
    """
    Prepares data for an attack by merging and deduplicating datasets.

    Args:
        model_indices: List of model indices over which to iterate and for which to gather information.
        model_type: Name of the model type for which we're loading data.
        models_base_dir: Where the various models' data lives.
        columns_for_deduplication: Names of columns to use in de-duplicating the dataframes

    Raises:
        ValueError: Throws if the list of model indices is empty.
        ValueError: Throws if any of the dataframes to be de-duplicated do not have the specified columns in
            ``columns_for_deduplication``

    Returns:
        A DataFrame containing the merged trainig data that has been deduplicated and is free from challenge data.
    """
    if len(model_indices) == 0:
        raise ValueError("The 'indices' list is empty. Please provide indices to process datasets.")

    df_merge_list = []
    df_challenge_list = []

    for model_index in model_indices:
        base_path = models_base_dir / f"{model_type}_{model_index}"
        df_merge_list.append(pd.read_csv(os.path.join(base_path, "train_with_id.csv")))
        df_challenge_list.append(pd.read_csv(os.path.join(base_path, "challenge_with_id.csv")))

    df_merge = pd.concat(df_merge_list, ignore_index=True)
    df_challenge = pd.concat(df_challenge_list, ignore_index=True)
    # Deduplicate the datasets once
    df_merge = df_merge.drop_duplicates(subset=columns_for_deduplication)
    df_challenge = df_challenge.drop_duplicates(subset=columns_for_deduplication)

    # Ensure all keys for deduplication exist in both DataFrames
    missing_keys_merge = [key for key in columns_for_deduplication if key not in df_merge.columns]
    missing_keys_challenge = [key for key in columns_for_deduplication if key not in df_challenge.columns]
    if missing_keys_merge or missing_keys_challenge:
        raise ValueError(
            f"Missing columns for deduplication in training data: {missing_keys_merge}"
            + f" and in challenge data: {missing_keys_challenge}"
        )

    # Remove challenge entries from the merged dataset
    return df_merge[
        ~df_merge.set_index(columns_for_deduplication).index.isin(
            df_challenge.set_index(columns_for_deduplication).index
        )
    ]


def run_data_processing(config: dict[str, Any]) -> None:
    """
    Run the data processing pipeline for the Tartan–Federer attack example.
    This function prepares the population datasets required for training and validating the attack.

    Args:
        config: Attack configuration as an OmegaConf DictConfig object.
    """
    log(INFO, "Running data processing pipeline...")

    population_data_path = Path(config["data_paths"]["population_data_path"])
    population_data_path.mkdir(parents=True, exist_ok=True)

    population_data_for_training_attack = prepare_population_dataset_for_attack(
        model_indices=config["data_processing_config"]["population_attack_indices_to_collect_for_training"],
        model_type=config["data_processing_config"]["model_type"],
        models_base_dir=Path(config["data_paths"]["midst_data_path"]),
        columns_for_deduplication=config["data_processing_config"]["columns_for_deduplication"],
    )

    population_data_for_training_attack.to_csv(
        population_data_path / "population_dataset_for_training_attack.csv",
        index=False,
    )

    population_data_for_validating_attack = prepare_population_dataset_for_attack(
        model_indices=config["data_processing_config"]["population_attack_indices_to_collect_for_validation"],
        model_type=config["data_processing_config"]["model_type"],
        models_base_dir=Path(config["data_paths"]["midst_data_path"]),
        columns_for_deduplication=config["data_processing_config"]["columns_for_deduplication"],
    )

    population_data_for_validating_attack.to_csv(
        population_data_path / "population_dataset_for_validating_attack.csv",
        index=False,
    )

    log(INFO, "Data processing pipeline finished.")


@hydra.main(config_path="configs", config_name="experiment_config", version_base=None)
def run_attack(config: DictConfig) -> None:
    """
    Run the Tartan–Federer attack example pipeline.

    Args:
        config: Attack configuration as an OmegaConf DictConfig object.
    """
    log(INFO, "Running Tartan–Federer attack...")

    set_all_random_seeds(
        seed=133742,
        use_deterministic_torch_algos=True,
        disable_torch_benchmarking=True,
    )

    cfg = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))

    if config["pipeline"]["run_data_processing"]:
        run_data_processing(cfg)

    data_cfg = cfg["data_paths"]
    attack_cfg = cfg["attack_config"]
    classifier_cfg = cfg["classifier_config"]

    mia_performance_train, mia_performance_val, mia_performance_test = tartan_federer_attack(
        model_type=attack_cfg["model_type"],
        model_data_dir=Path(attack_cfg["models_base_dir"]),
        target_model_subdir=Path(attack_cfg["target_shadow_model_subdir"]),
        samples_per_train_model=attack_cfg["samples_per_train_model"],
        sample_per_val_model=attack_cfg["samples_per_val_model"],
        num_noise_per_time_step=attack_cfg["num_noise_per_time_step"],
        timesteps=attack_cfg["timesteps"],
        additional_timesteps=attack_cfg["additional_timesteps"],
        predictions_file_format=attack_cfg["predictions_file_name"],
        results_path=Path(attack_cfg["results_path"]),
        test_indices=attack_cfg["test_indices"],
        train_indices=attack_cfg["train_indices"],
        val_indices=attack_cfg["val_indices"],
        columns_for_deduplication=attack_cfg["columns_for_deduplication"],
        classifier_hidden_dim=classifier_cfg["hidden_dim"],
        classifier_num_epochs=classifier_cfg["num_epochs"],
        classifier_learning_rate=classifier_cfg["learning_rate"],
        meta_dir=Path(config["data_paths"]["metadata_dir"]),
        population_data_dir=Path(data_cfg["population_data_path"]),
    )

    unset_all_random_seeds()

    log(INFO, "Attack finished successfully.")


if __name__ == "__main__":
    run_attack()
