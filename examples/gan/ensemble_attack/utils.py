from typing import Any
from pathlib import Path
import json

from omegaconf import DictConfig, OmegaConf

from midst_toolkit.attacks.ensemble.shadow_model_utils import setup_save_dir
from midst_toolkit.attacks.ensemble.model import EnsembleAttackCTGANTrainingConfig


def make_training_config(config: DictConfig) -> EnsembleAttackCTGANTrainingConfig:
    # Saving the model config from the config.yaml into a json file
    # because that's what the ensemble attack code will be looking for
    training_config_path = Path(config.ensemble_attack.shadow_training.training_json_config_paths.training_config_path)
    training_config_path.unlink(missing_ok=True)
    with open(training_config_path, "w") as f:
        training_config = OmegaConf.to_container(config.ensemble_attack.shadow_training.model_config, resolve=True)
        assert isinstance(training_config, dict), "Training config must be a dictionary."
        training_config["general"] = {
            "test_data_dir": config.base_data_dir,
            "sample_prefix": "ctgan",
            "data_dir": config.base_data_dir,
            "workspace_dir": str(Path(config.base_data_dir) / "shadow_workspace"),
            "exp_name": "pre_trained_model",
        }
        json.dump(training_config, f)

    ctgan_training_config = EnsembleAttackCTGANTrainingConfig(**training_config)

    setup_save_dir(ctgan_training_config)

    return ctgan_training_config
