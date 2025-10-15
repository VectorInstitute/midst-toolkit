import json
from pathlib import Path

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from midst_toolkit.attacks.ensemble.shadow_model_utils import (
    save_additional_tabddpm_config,
)


@pytest.fixture(scope="module")
def cfg() -> DictConfig:
    with initialize(config_path="configs"):
        return compose(config_name="shadow_training_config")


def test_save_additional_tabddpm_config(cfg: DictConfig, tmp_path: Path) -> None:
    # Input path
    tabddpm_config_path = Path(cfg.shadow_training.training_json_config_paths.tabddpm_training_config_path)

    # Extract original parameters
    with open(tabddpm_config_path, "r") as file:
        config_data = json.load(file)
    old_data_dir = config_data["general"]["data_dir"]
    old_workspace_dir = config_data["general"]["workspace_dir"]
    old_exp_name = config_data["general"]["exp_name"]

    # New parameters
    new_data_dir = tmp_path / "data_dir"
    new_workspace_name = "test_workspace"
    new_experiment_name = "test_experiment"
    final_json_path = tmp_path / "modified_config.json"

    configs, save_dir = save_additional_tabddpm_config(
        data_dir=new_data_dir,
        training_config_json_path=tabddpm_config_path,
        final_config_json_path=final_json_path,
        experiment_name=new_experiment_name,
        workspace_name=new_workspace_name,
    )

    assert save_dir == new_data_dir / new_workspace_name / new_experiment_name
    assert configs["general"]["data_dir"] == str(new_data_dir)
    assert configs["general"]["workspace_dir"] == str(new_data_dir / new_workspace_name)
    assert configs["general"]["exp_name"] == new_experiment_name
    # Ensure original parameters are different from new ones
    assert old_data_dir != configs["general"]["data_dir"]
    assert old_workspace_dir != configs["general"]["workspace_dir"]
    assert old_exp_name != configs["general"]["exp_name"]
    # Ensure required directories are created
    assert (save_dir / "models").exists()
    assert (save_dir / "before_matching").exists()
