"""
This file is an uncompleted example script for running the Ensemble Attack on MIDST challenge
provided resources and data.
"""

from logging import INFO
from pathlib import Path

import hydra
from omegaconf import DictConfig

import examples.ensemble_attack.berka.run_metaclassifier_training as meta_pipeline
import examples.ensemble_attack.berka.run_shadow_model_training as shadow_pipeline
from examples.ensemble_attack.berka.real_data_collection import collect_population_data_ensemble
from midst_toolkit.attacks.ensemble.data_utils import load_dataframe
from midst_toolkit.attacks.ensemble.process_split_data import process_split_data
from midst_toolkit.common.logger import log
from midst_toolkit.common.random import set_all_random_seeds


def run_data_processing(config: DictConfig) -> None:
    """
    Function to run the data processing pipeline.

    Args:
        config: Configuration object set in config.yaml.
    """
    # Load original repo's population to be concatenated to the experiment's population data.
    # Original population refers to the population data collected by the attacker team, and can
    # be downloaded from: https://github.com/CRCHUM-CITADEL/ensemble-mia/blob/main/input/population/population_all_with_challenge.csv
    # This concatenation is done to align the experiments with the original attack code because
    # this attack needs a large population dataset, and only using the experiment's collected population
    # is not enough.
    original_population_data = load_dataframe(
        Path(config.data_processing_config.original_population_data_path),
        "population_all_with_challenge.csv",
    )
    log(INFO, "Running data processing pipeline...")
    # Collect the real data from the MIDST challenge resources.
    population_data = collect_population_data_ensemble(
        midst_data_input_dir=Path(config.data_processing_config.midst_data_path),
        data_processing_config=config.data_processing_config,
        save_dir=Path(config.data_paths.population_path),
        base_population=original_population_data,
        population_splits=config.data_processing_config.population_splits,
        challenge_splits=config.data_processing_config.challenge_splits,
    )

    # The following function saves the required dataframe splits in the specified processed_attack_data_path path.
    process_split_data(
        all_population_data=population_data,
        processed_attack_data_path=Path(config.data_paths.processed_attack_data_path),
        # TODO: column_to_stratify value is not documented in the original codebase.
        column_to_stratify=config.data_processing_config.column_to_stratify,
        num_total_samples=config.data_processing_config.population_sample_size,
        random_seed=config.random_seed,
    )
    log(INFO, "Data processing pipeline finished.")


@hydra.main(config_path="configs", config_name="experiment_config", version_base=None)
def main(config: DictConfig) -> None:
    """
    Run the Ensemble Attack example pipeline.
    As the first step, data processing is done.
    Second step is shadow model training used for RMIA attack.
    Third step is metaclassifier training and evaluation.

    Args:
        config: Attack configuration as an OmegaConf DictConfig object.
    """
    if config.random_seed is not None:
        set_all_random_seeds(seed=config.random_seed)
        log(INFO, f"Training phase random seed set to {config.random_seed}.")

    if config.pipeline.run_data_processing:
        run_data_processing(config)

    if config.pipeline.run_shadow_model_training:
        df_master_challenge_train = load_dataframe(
            Path(config.data_paths.processed_attack_data_path),
            "master_challenge_train.csv",
        )
        shadow_data_paths = shadow_pipeline.run_shadow_model_training(config, df_master_challenge_train)
        shadow_data_paths = [Path(path) for path in shadow_data_paths]

        target_model_synthetic_path = shadow_pipeline.run_target_model_training(config)

    if config.pipeline.run_metaclassifier_training:
        if not config.pipeline.run_shadow_model_training:
            # If shadow model training is skipped, we need to provide the previous shadow model and target model paths.
            shadow_data_paths = [Path(path) for path in config.shadow_training.final_shadow_models_path]
            target_model_synthetic_path = Path(config.shadow_training.target_synthetic_data_path)

        assert len(shadow_data_paths) == 3, "The attack_data_paths list must contain exactly three elements."
        assert target_model_synthetic_path is not None, (
            "The target_data_path must be provided for metaclassifier training."
        )

        meta_pipeline.run_metaclassifier_training(config, shadow_data_paths, target_model_synthetic_path)


if __name__ == "__main__":
    main()
