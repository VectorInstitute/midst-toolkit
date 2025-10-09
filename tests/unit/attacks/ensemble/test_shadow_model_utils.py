import copy
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from midst_toolkit.attacks.ensemble.shadow_model_utils import (
    config_tabddpm,
    fine_tune_tabddpm_and_synthesize,
    train_tabddpm_and_synthesize,
)


@pytest.fixture(scope="module")
def cfg() -> DictConfig:
    with initialize(config_path="configs"):
        return compose(config_name="shadow_training_config")


def test_config_tabddpm(cfg: DictConfig, tmp_path: Path) -> None:
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

    configs, save_dir = config_tabddpm(
        data_dir=new_data_dir,
        training_json_path=tabddpm_config_path,
        final_json_path=final_json_path,
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
    assert Path(save_dir / "models").exists()
    assert Path(save_dir / "before_matching").exists()


def test_train_and_fine_tune_tabddpm(cfg: DictConfig, tmp_path: Path) -> None:
    # Input
    train_set = pd.read_csv(
        "tests/unit/attacks/ensemble/assets/population_data/all_population.csv"
    )  # For testing purposes only.
    fine_tuning_set = copy.deepcopy(train_set)
    tabddpm_config_path = Path(cfg.shadow_training.training_json_config_paths.tabddpm_training_config_path)
    tmp_training_dir = tmp_path
    # We should move ``dataset_meta.json`` and ``trans_domain.json`` files to the ``tmp_training_dir``
    assert Path(cfg.shadow_training.training_json_config_paths.table_domain_file_path).exists()
    shutil.copyfile(
        cfg.shadow_training.training_json_config_paths.table_domain_file_path,
        Path(tmp_training_dir / "trans_domain.json"),
    )

    shutil.copyfile(
        cfg.shadow_training.training_json_config_paths.dataset_meta_file_path,
        Path(tmp_training_dir / "dataset_meta.json"),
    )
    configs, save_dir = config_tabddpm(
        data_dir=tmp_training_dir,
        training_json_path=tabddpm_config_path,
        final_json_path=tmp_training_dir / "trans.json",
        experiment_name="test_experiment",
        workspace_name="test_workspace",
    )

    train_result = train_tabddpm_and_synthesize(
        train_set,
        configs,
        save_dir,
        synthesize=True,
    )
    # By default, with a sampling scale of 1, the size of the synthesized data is equal
    # to the size of the training data.
    assert "synth_data" in train_result
    assert type(train_result["synth_data"]) is pd.DataFrame
    assert len(train_result["synth_data"]) == 99

    assert "models" in train_result
    assert type(train_result["models"]) is dict
    assert len(train_result["models"]) == 1  # Only one model (TabDDPM) is trained.

    # Now fine-tune the trained TabDDPM model on a small set of data
    fine_tuned_results = fine_tune_tabddpm_and_synthesize(
        trained_models=train_result["models"],
        new_train_set=fine_tuning_set,  # fine-tuning on the same data for testing purposes
        configs=configs,
        save_dir=save_dir,
        fine_tuning_diffusion_iterations=cfg.shadow_training.fine_tuning_config.fine_tune_diffusion_iterations,
        fine_tuning_classifier_iterations=cfg.shadow_training.fine_tuning_config.fine_tune_classifier_iterations,
        # Number of synthetic samples is defined according to tabddpm_training_config's classifier_scale value.
        synthesize=False,
    )
    assert "synth_data" in fine_tuned_results
    assert fine_tuned_results["synth_data"] == {}
    assert "new_models" in fine_tuned_results
    assert type(fine_tuned_results["new_models"]) is dict
    assert len(fine_tuned_results["new_models"]) == 1  # Only one model (TabDDPM) is fine-tuned.
