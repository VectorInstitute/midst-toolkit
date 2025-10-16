"""
This file is an uncompleted example script for running the Ensemble Attack on MIDST challenge
provided resources and data.
"""

import importlib
from logging import INFO
from pathlib import Path

import hydra
from omegaconf import DictConfig

from examples.ensemble_attack.real_data_collection import collect_population_data_ensemble
from midst_toolkit.attacks.ensemble.process_split_data import process_split_data
from midst_toolkit.common.logger import log


def run_data_processing(config: DictConfig) -> None:
    """
    Function to run the data processing pipeline.

    Args:
        config: Configuration object set in config.yaml.
    """
    log(INFO, "Running data processing pipeline...")
    # Collect the real data from the MIDST challenge resources.
    population_data = collect_population_data_ensemble(
        midst_data_input_dir=Path(config.data_paths.midst_data_path),
        data_processing_config=config.data_processing_config,
        save_dir=Path(config.data_paths.population_path),
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


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """
    Run the Ensemble Attack example pipeline.
    As the first step, data processing is done.
    Second step is shadow model training used for RMIA attack.
    Third step is metaclassifier training and evaluation.

    Args:
        config: Attack configuration as an OmegaConf DictConfig object.
    """
    if config.pipeline.run_data_processing:
        run_data_processing(config)

    # Note: Importing the following two modules causes a segmentation fault error if imported together in this file.
    # A quick solution is to load modules dynamically if any of the pipelines is called.
    # TODO: Investigate the source of error.
    if config.pipeline.run_shadow_model_training:
        shadow_pipeline = importlib.import_module("examples.ensemble_attack.run_shadow_model_training")
        attack_data_paths = shadow_pipeline.run_shadow_model_training(config)

    else:
        # If shadow model training is skipped, we need to provide the paths to pre-trained shadow models.
        attack_data_paths = [
            "initial_model_rmia_1/shadow_workspace/pre_trained_model/rmia_shadows.pkl",
            "initial_model_rmia_2/shadow_workspace/pre_trained_model/rmia_shadows.pkl",
            "shadow_model_rmia_third_set/shadow_workspace/trained_model/rmia_shadows_third_set.pkl",
        ]

    if config.pipeline.run_metaclassifier_training:
        meta_pipeline = importlib.import_module("examples.ensemble_attack.run_metaclassifier_training")
        meta_pipeline.run_metaclassifier_training(config, attack_data_paths, target_data_path="")


if __name__ == "__main__":
    main()
