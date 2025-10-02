import pickle
from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from midst_toolkit.attacks.ensemble.data_utils import load_dataframe
from midst_toolkit.attacks.ensemble.rmia.shadow_model_training import (
    train_fine_tuning_shadows,
    train_shadow_on_half_challenge_data,
)


@pytest.fixture(scope="module")
def cfg() -> DictConfig:
    with initialize(config_path="configs"):
        return compose(config_name="shadow_training_config")


def test_train_fine_tuning_shadows(cfg: DictConfig, tmp_path: Path) -> None:
    # Models and training artifacts are saved under ``shadow_models_data_path``
    # Replace ``tmp_path`` with config's ``shadow_models_data_path`` in the next line to save
    # the trained models at and see the output files.
    shadow_models_data_path = tmp_path
    # Input
    # Population data is used to pre-train some of the shadow models.
    population_data = load_dataframe(Path("tests/unit/attacks/ensemble/assets/population_data"), "all_population.csv")
    result_path = train_fine_tuning_shadows(
        n_models=2,
        n_reps=1,
        population_data=population_data,
        master_challenge_data=population_data[0:20],  # For testing purposes only.
        shadow_models_data_path=shadow_models_data_path,
        training_json_config_paths=cfg.shadow_training.training_json_config_paths,
        fine_tuning_config=cfg.shadow_training.fine_tuning_config,
        init_model_id=1,
        init_data_seed=cfg.random_seed,
        pre_training_data_size=cfg.shadow_training.fine_tuning_config.pre_train_data_size,
        random_seed=cfg.random_seed,
    )
    # Expected saved models and synthesized data:
    # Load the saved pickle
    assert result_path.exists(), f"Result path {result_path} is not created."
    with open(result_path, "rb") as file:
        shadow_data = pickle.load(file)

    assert len(shadow_data["fine_tuning_sets"]) == 2  # n_models
    assert len(shadow_data["fine_tuned_results"]) == 2  # n_models
    for result in shadow_data["fine_tuned_results"]:
        assert "synth_data" in result
        assert "tables" in result
        assert "new_models" in result
        assert "configs" in result
        assert "save_dir" in result
        assert "relation_order" in result
        assert "all_group_lengths_prob_dicts" in result
        assert type(result["synth_data"]) is pd.DataFrame

    # Fine tuning sets should be disjoint
    assert set(shadow_data["fine_tuning_sets"][0]).isdisjoint(set(shadow_data["fine_tuning_sets"][1]))
    # Fine tuning sets should be unique
    assert len(shadow_data["fine_tuning_sets"][0]) == len(set(shadow_data["fine_tuning_sets"][0]))
    assert len(shadow_data["fine_tuning_sets"][1]) == len(set(shadow_data["fine_tuning_sets"][1]))


def test_train_shadow_on_half_challenge_data(cfg: DictConfig, tmp_path: Path) -> None:
    # Models and training artifacts are saved under ``shadow_models_data_path``
    # Replace ``tmp_path`` with config's ``shadow_models_data_path`` in the next line to save
    # the trained models at and see the output files.
    shadow_models_data_path = tmp_path
    # Input
    # Population data is loaded and used as challenge data for testing purposes.
    population_data = load_dataframe(Path("tests/unit/attacks/ensemble/assets/population_data"), "all_population.csv")
    result_path = train_shadow_on_half_challenge_data(
        n_models=2,
        n_reps=1,
        master_challenge_data=population_data[0:40],  # For testing purposes only.
        shadow_models_data_path=shadow_models_data_path,
        training_json_config_paths=cfg.shadow_training.training_json_config_paths,
        random_seed=cfg.random_seed,
    )
    # Expected saved models and synthesized data:
    # Load the saved pickle
    assert result_path.exists(), f"Result path {result_path} is not created."
    with open(result_path, "rb") as file:
        shadow_data = pickle.load(file)

    assert len(shadow_data["selected_sets"]) == 2  # n_models
    assert len(shadow_data["trained_results"]) == 2  # n_models
    for result in shadow_data["trained_results"]:
        assert "synth_data" in result
        assert "tables" in result
        assert "models" in result
        assert "configs" in result
        assert "save_dir" in result
        assert "relation_order" in result
        assert "all_group_lengths_prob_dicts" in result
        assert type(result["synth_data"]) is pd.DataFrame

    # Training sets should be disjoint
    assert set(shadow_data["selected_sets"][0]).isdisjoint(set(shadow_data["selected_sets"][1]))
    # Training sets should be unique
    assert len(shadow_data["selected_sets"][0]) == len(set(shadow_data["selected_sets"][0]))
    assert len(shadow_data["selected_sets"][1]) == len(set(shadow_data["selected_sets"][1]))
