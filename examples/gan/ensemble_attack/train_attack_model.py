import json
from logging import INFO
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from examples.ensemble_attack.run_metaclassifier_training import run_metaclassifier_training
from examples.ensemble_attack.run_shadow_model_training import run_shadow_model_training, run_target_model_training
from examples.gan.utils import get_single_table_svd_metadata, get_table_name
from midst_toolkit.attacks.ensemble.data_utils import load_dataframe, save_dataframe
from midst_toolkit.attacks.ensemble.model import EnsembleAttackCTGANModelRunner, EnsembleAttackCTGANTrainingConfig
from midst_toolkit.attacks.ensemble.process_split_data import process_split_data
from midst_toolkit.attacks.ensemble.shadow_model_utils import save_additional_training_config
from midst_toolkit.common.logger import log
from midst_toolkit.common.random import set_all_random_seeds


@hydra.main(config_path="./", config_name="config", version_base=None)
def train_attack_model(config: DictConfig) -> None:
    """
    Train the Ensemble Attack pipeline with CTGAN model.

    As the first step, data processing is done.
    Second step is shadow model training used for RMIA attack.
    Third step is metaclassifier training and evaluation.

    Args:
        config: Attack configuration as an OmegaConf DictConfig object.
    """
    if config.ensemble_attack.random_seed is not None:
        set_all_random_seeds(seed=config.ensemble_attack.random_seed)
        log(INFO, f"Training phase random seed set to {config.ensemble_attack.random_seed}.")

    if config.ensemble_attack.pipeline.run_data_processing:
        log(INFO, "Running data processing pipeline...")
        # The following function saves the required dataframe splits in the specified processed_attack_data_path path.
        population_data = load_dataframe(
            Path(config.ensemble_attack.data_paths.population_path),
            config.data_file_name,
        )

        # Removing id columns and saving the dataset
        id_columns = [c for c in population_data.columns if c.endswith("_id")]
        population_data_no_id = population_data.drop(columns=id_columns)
        save_dataframe(
            population_data_no_id,
            Path(config.ensemble_attack.data_paths.population_path),
            f"{Path(config.data_file_name).stem}_no_id.csv",
        )

        process_split_data(
            all_population_data=population_data,
            processed_attack_data_path=Path(config.ensemble_attack.data_paths.processed_attack_data_path),
            # TODO: column_to_stratify value is not documented in the original codebase.
            column_to_stratify=config.ensemble_attack.data_processing_config.column_to_stratify,
            num_total_samples=config.ensemble_attack.data_processing_config.population_sample_size,
            random_seed=config.ensemble_attack.random_seed,
        )

    # Saving the model config from the config.yaml into a json file
    # because that's what the ensemble attack code will be looking for
    training_config_path = Path(config.ensemble_attack.shadow_training.training_json_config_paths.training_config_path)
    training_config_path.unlink(missing_ok=True)
    with open(training_config_path, "w") as f:
        training_config = OmegaConf.to_container(config.ensemble_attack.shadow_training.model_config)
        assert isinstance(training_config, dict), "Training config must be a dictionary."
        training_config["general"] = {
            "test_data_dir": config.base_data_dir,
            "sample_prefix": "ctgan",
            # The values below will be overriden
            "exp_name": "",
            "data_dir": "",
            "workspace_dir": "",
        }
        json.dump(training_config, f)

    if config.ensemble_attack.pipeline.run_shadow_model_training:
        log(INFO, "Training the shadow models...")
        master_challenge_train = load_dataframe(
            Path(config.ensemble_attack.data_paths.population_path),
            "master_challenge_train.csv",
        )

        table_name = get_table_name(config.base_data_dir)
        domain_file_path = Path(config.base_data_dir) / f"{table_name}_domain.json"
        with open(domain_file_path, "r") as file:
            domain_dictionary = json.load(file)

        training_config, _ = save_additional_training_config(
            training_config_type=EnsembleAttackCTGANTrainingConfig,
            data_dir=Path(config.base_data_dir),
            training_config_json_path=training_config_path,
            final_config_json_path=Path(config.base_data_dir) / f"{table_name}.json",  # Path to the new json
            experiment_name="pre_trained_model",
        )

        metadata, _ = get_single_table_svd_metadata(master_challenge_train, domain_dictionary)
        training_config.metadata = metadata
        training_config.table_name = table_name

        model_runner = EnsembleAttackCTGANModelRunner(training_config=training_config)

        shadow_data_paths = run_shadow_model_training(model_runner, config.ensemble_attack, master_challenge_train)
        shadow_data_paths = [Path(path) for path in shadow_data_paths]

        log(INFO, "Training the target model...")
        target_model_synthetic_path = run_target_model_training(config.ensemble_attack)

    if config.ensemble_attack.pipeline.run_metaclassifier_training:
        log(INFO, "Training the metaclassifier...")
        if not config.ensemble_attack.pipeline.run_shadow_model_training:
            # If shadow model training is skipped, we need to provide the previous shadow model and target model paths.
            shadow_data_paths = [
                Path(path) for path in config.ensemble_attack.shadow_training.final_shadow_models_path
            ]
            target_model_synthetic_path = Path(config.ensemble_attack.shadow_training.target_synthetic_data_path)

        assert len(shadow_data_paths) == 3, "The attack_data_paths list must contain exactly three elements."
        assert target_model_synthetic_path is not None, (
            "The target_data_path must be provided for metaclassifier training."
        )

        run_metaclassifier_training(config.ensemble_attack, shadow_data_paths, target_model_synthetic_path)


if __name__ == "__main__":
    train_attack_model()
