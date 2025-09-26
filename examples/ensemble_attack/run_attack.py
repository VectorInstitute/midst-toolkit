"""
This file is an uncompleted example script for running the Ensemble Attack on MIDST challenge
provided resources and data.
"""

from logging import INFO
from pathlib import Path

import hydra
from omegaconf import DictConfig

from examples.ensemble_attack.real_data_collection import collect_population_data_ensemble
from midst_toolkit.attacks.ensemble.data_utils import load_dataframe
from midst_toolkit.attacks.ensemble.process_split_data import process_split_data
from midst_toolkit.attacks.ensemble.rmia.shadow_model_training import (
    run_shadow_model_training,
)
from midst_toolkit.common.logger import log


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Run the Ensemble Attack example pipeline.
    As the first step, data processing is done.
    Second step is shadow model training used for RMIA attack.

    Args:
        cfg: Attack OmegaConf DictConfig object.
    """
    if cfg.pipeline.run_data_processing:
        log(INFO, "Running data processing pipeline...")
        # Collect the real data from the MIDST challenge resources.
        population_data = collect_population_data_ensemble(
            midst_data_input_dir=Path(cfg.data_paths.midst_data_path),
            data_processing_config=cfg.data_processing_config,
            save_dir=Path(cfg.data_paths.population_path),
        )
        # The following function saves the required dataframe splits in the specified processed_attack_data_path path.
        (
            df_real_train,
            df_real_val,
            df_real_test,
            df_master_challenge_train,
            df_master_challenge_test,
        ) = process_split_data(
            all_population_data=population_data,
            processed_attack_data_path=Path(cfg.data_paths.processed_attack_data_path),
            # TODO: column_to_stratify value is not documented in the original codebase.
            column_to_stratify=cfg.data_processing_config.column_to_stratify,
            num_total_samples=cfg.data_processing_config.population_sample_size,
            random_seed=cfg.random_seed,
        )
        log(INFO, "Data processing pipeline finished.")

    if cfg.pipeline.run_shadow_model_training:
        log(INFO, "Running shadow model training...")
        # Load the required dataframes for shadow model training.
        # For shadow model training we need master_challenge_train and population data.
        # Master challenge is the main training (or fine-tuning) data for the shadow models.
        df_master_challenge_train = load_dataframe(
            Path(cfg.data_paths.processed_attack_data_path),
            "master_challenge_train.csv",
        )
        # Population data is used to pre-train some of the shadow models.
        df_population_with_challenge = load_dataframe(
            Path(cfg.data_paths.population_path), "population_all_with_challenge.csv"
        )
        # ``population_data`` in ensemble attack is often used for shadow pre-training, and
        # ``master_challenge_df`` is used for fine-tuning.
        run_shadow_model_training(
            population_data=df_population_with_challenge,
            master_challenge_data=df_master_challenge_train,
            shadow_models_data_path=Path(cfg.shadow_training.shadow_models_data_path),
            training_json_config_paths=cfg.shadow_training.training_json_config_paths,
            shadow_training_config=cfg.shadow_training,
            n_models=2,  # 4 based on the original code, must be even
            n_reps=12,  # 12 based on the original code
            random_seed=cfg.random_seed,
        )


if __name__ == "__main__":
    main()
