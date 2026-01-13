import importlib
import json
from logging import INFO
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from examples.ensemble_attack.run_shadow_model_training import run_shadow_model_training, run_target_model_training
from midst_toolkit.common.logger import log
from midst_toolkit.common.random import set_all_random_seeds


@hydra.main(config_path="./", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """
    Run the Ensemble Attack pipeline with the CTGAN model.

    As the first step, data processing is done.
    Second step is shadow model training used for RMIA attack.
    Third step is metaclassifier training and evaluation.

    Args:
        config: Attack configuration as an OmegaConf DictConfig object.
    """
    if config.ensemble_attack.random_seed is not None:
        set_all_random_seeds(seed=config.ensemble_attack.random_seed)
        log(INFO, f"Training phase random seed set to {config.ensemble_attack.random_seed}.")

    # Saving the model config from the config.yaml into a json file
    # because that's what the ensemble attack code will be looking for
    training_config_path = Path(config.ensemble_attack.shadow_training.training_json_config_paths.training_config_path)
    training_config_path.unlink(missing_ok=True)
    with open(training_config_path, "w") as f:
        training_config = OmegaConf.to_container(config.ensemble_attack.shadow_training.model_config)
        training_config["general"] = {
            "test_data_dir": config.base_data_dir,
            "sample_prefix": "ctgan",
            # The values below will be overriden
            "exp_name": "",
            "data_dir": "",
            "workspace_dir": "",
        }
        json.dump(training_config, f)

    shadow_data_paths = run_shadow_model_training(config.ensemble_attack)
    shadow_data_paths = [Path(path) for path in shadow_data_paths]

    target_model_synthetic_path = run_target_model_training(config)

    if config.pipeline.run_metaclassifier_training:
        if not config.pipeline.run_shadow_model_training:
            # If shadow model training is skipped, we need to provide the previous shadow model and target model paths.
            shadow_data_paths = [Path(path) for path in config.shadow_training.final_shadow_models_path]
            target_model_synthetic_path = Path(config.shadow_training.target_synthetic_data_path)

        assert len(shadow_data_paths) == 3, "The attack_data_paths list must contain exactly three elements."
        assert target_model_synthetic_path is not None, (
            "The target_data_path must be provided for metaclassifier training."
        )

        meta_pipeline = importlib.import_module("examples.ensemble_attack.run_metaclassifier_training")
        meta_pipeline.run_metaclassifier_training(config, shadow_data_paths, target_model_synthetic_path)


if __name__ == "__main__":
    main()
