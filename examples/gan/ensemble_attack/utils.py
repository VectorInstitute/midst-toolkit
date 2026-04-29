import json
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from midst_toolkit.attacks.ensemble.data_utils import load_dataframe
from midst_toolkit.attacks.ensemble.process_split_data import PROCESSED_TRAIN_DATA_FILE_NAME
from midst_toolkit.attacks.ensemble.shadow_model_utils import setup_save_dir


def get_master_challenge_train_data(config: DictConfig) -> pd.DataFrame:
    """
    Get the master challenge train data from the config's population path location.

    Args:
        config: The configuration object.

    Returns:
        The dataframe containing the master challenge train data.
    """
    population_path = Path(config.ensemble_attack.data_paths.population_path)
    assert population_path.exists(), (
        f"Population path {population_path} does not exist. Please run the data processing pipeline first."
    )

    return load_dataframe(population_path, PROCESSED_TRAIN_DATA_FILE_NAME)


def make_training_config(config: DictConfig) -> dict[Any, Any]:
    """
    Make the ensemble attack training config for the CTGAN model from the config.yaml file.

    Saves the training config json file to the shadow training json config paths location.

    Args:
        config: The configuration object.

    Returns:
        The ensemble attack training config for the CTGAN model.
    """
    base_data_dir = str
    if "base_data_dir" in config:
        base_data_dir = config.base_data_dir
    if "data_dir" in config:
        base_data_dir = config.data_dir
    else:
        raise ValueError("Either base_data_dir or data_dir must be provided in the config.")

    # Saving the model config from the config.yaml into a json file
    # because that's what the ensemble attack code will be looking for
    training_config_path = Path(config.ensemble_attack.shadow_training.training_json_config_paths.training_config_path)
    training_config_path.unlink(missing_ok=True)
    with open(training_config_path, "w") as f:
        training_config = OmegaConf.to_container(config.ensemble_attack.shadow_training.model_config, resolve=True)
        assert isinstance(training_config, dict), "Training config must be a dictionary."
        training_config["general"] = {
            "test_data_dir": base_data_dir,
            "sample_prefix": "ctgan",
            "data_dir": base_data_dir,
            "workspace_dir": str(Path(base_data_dir) / "shadow_workspace"),
            "exp_name": "pre_trained_model",
        }
        json.dump(training_config, f)

    setup_save_dir(training_config)

    return training_config
