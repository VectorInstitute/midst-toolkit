import json
import os
from logging import INFO
from pathlib import Path
from typing import Any

from midst_toolkit.common.config import TrainingConfig
from midst_toolkit.common.logger import log


def update_and_save_training_config(
    config: TrainingConfig,
    data_dir: Path,
    final_config_json_path: Path,
    experiment_name: str = "attack_experiment",
    workspace_name: str = "shadow_workspace",
) -> TrainingConfig:
    """
    Modifies a model configuration with the specified data directory, experiment name and workspace name,
    and saves it to a JSON file.

    Args:
        config: The training configuration to update.
        data_dir: Directory containing dataset_meta.json, trans_domain.json, and trans.json files.
        final_config_json_path: Path where the modified configuration JSON file will be saved.
        experiment_name: Name of the experiment, used to create a unique save directory.
        workspace_name: Name of the workspace, used to create a unique save directory.

    Returns:
        EnsembleAttackTrainingConfig: The updated training configuration
    """
    # Modify the config to have the correct training data and saving directory
    config.general.data_dir = data_dir
    # Save dir is set by joining the workspace_dir and exp_name
    config.general.workspace_dir = data_dir / workspace_name
    config.general.exp_name = experiment_name

    # save the changed to the new json file
    with open(final_config_json_path, "w") as file:
        json.dump(config.model_dump(mode="json"), file, indent=4)

    log(INFO, f"Config saved to {final_config_json_path}")

    # Set up the config
    config.save_dir = setup_save_dir(config.model_dump(mode="json"))

    return config


# TODO: Move this code to ClavaDDPM since it's probably only needed there.
# The following function is directly copied from the midst reference code since
# I need it to run the attack code, but, it should probably be moved to somewhere else
# as it is an essential part of a working TabDDPM training pipeline.
def setup_save_dir(configs: dict[Any, Any]) -> Path:
    """
    Set up the directories where the models and intermediate results will be saved.

    The following directories are created:
        - save_dir -> configs.general.workspace_dir/configs.general.exp_name
        - save_dir/models
        - save_dir/before_matching

    Additionally, a json file with the configuration settings is saved to ``save_dir/args``.

    Args:
        configs: Configuration settings.

    Returns:
        save_dir: Directory path where results will be saved.
    """
    # Following directories are created to save the models and intermediate results.
    save_dir = Path(configs["general"]["workspace_dir"]) / Path(configs["general"]["exp_name"])
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir / "models", exist_ok=True)
    os.makedirs(save_dir / "before_matching", exist_ok=True)

    with open(save_dir / "args", "w") as file:
        json.dump(configs, file, indent=4)

    return save_dir
