import copy
import json
import pickle
import shutil
from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from midst_toolkit.attacks.ensemble.data_utils import load_dataframe
from midst_toolkit.attacks.ensemble.models import EnsembleAttackClavaDDPMModelRunner
from midst_toolkit.attacks.ensemble.rmia.shadow_model_training import (
    train_fine_tuned_shadow_models,
    train_shadow_on_half_challenge_data,
)
from midst_toolkit.attacks.ensemble.shadow_model_utils import update_and_save_training_config
from midst_toolkit.common.config import ClavaDDPMTrainingConfig


POPULATION_DATA = load_dataframe(
    Path("tests/integration/attacks/ensemble/assets/population_data"),
    "all_population.csv",
)


@pytest.fixture(scope="module")
def cfg() -> DictConfig:
    with initialize(config_path="configs"):
        return compose(config_name="shadow_training_config")


@pytest.mark.integration_test()
def test_train_fine_tuned_shadow_models(cfg: DictConfig, tmp_path: Path) -> None:
    # Models and training artifacts are saved under ``shadow_models_output_path``
    # Replace ``tmp_path`` with config's ``shadow_models_output_path`` in the next line to save
    # the trained models at and see the output files.
    shadow_models_output_path = tmp_path
    # Input
    # Population data is used to pre-train some of the shadow models.
    model_runner = EnsembleAttackClavaDDPMModelRunner(cfg)
    result_path = train_fine_tuned_shadow_models(
        model_runner=model_runner,
        n_models=2,
        n_reps=1,
        population_data=POPULATION_DATA,
        master_challenge_data=POPULATION_DATA[0:20],  # Limiting the data to 20 samples for faster test execution
        shadow_models_output_path=shadow_models_output_path,
        training_json_config_paths=cfg.shadow_training.training_json_config_paths,
        init_model_id=1,
        init_data_seed=cfg.random_seed,
        table_name="trans",
        id_column_name="trans_id",
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
    for synthetic_data in shadow_data["fine_tuned_results"]:
        assert type(synthetic_data) is pd.DataFrame
        assert synthetic_data is not None
        assert len(synthetic_data) == 5

    # Fine tuning sets should be disjoint
    assert set(shadow_data["fine_tuning_sets"][0]).isdisjoint(set(shadow_data["fine_tuning_sets"][1]))
    # Fine tuning sets should be unique
    assert len(shadow_data["fine_tuning_sets"][0]) == len(set(shadow_data["fine_tuning_sets"][0]))
    assert len(shadow_data["fine_tuning_sets"][1]) == len(set(shadow_data["fine_tuning_sets"][1]))


@pytest.mark.integration_test()
def test_train_shadow_on_half_challenge_data(cfg: DictConfig, tmp_path: Path) -> None:
    # Models and training artifacts are saved under ``shadow_models_output_path``
    # Replace ``tmp_path`` with config's ``shadow_models_output_path`` in the next line to save
    # the trained models at and see the output files.
    shadow_models_output_path = tmp_path
    # Input
    # Population data is loaded and used as challenge data for testing purposes.
    model_runner = EnsembleAttackClavaDDPMModelRunner(cfg)
    result_path = train_shadow_on_half_challenge_data(
        model_runner=model_runner,
        n_models=2,
        n_reps=1,
        master_challenge_data=POPULATION_DATA[0:40],  # Limiting the data to 40 samples for faster test execution
        shadow_models_output_path=shadow_models_output_path,
        training_json_config_paths=cfg.shadow_training.training_json_config_paths,
        table_name="trans",
        id_column_name="trans_id",
        random_seed=cfg.random_seed,
    )
    # Expected saved models and synthesized data:
    # Load the saved pickle
    assert result_path.exists(), f"Result path {result_path} is not created."
    with open(result_path, "rb") as file:
        shadow_data = pickle.load(file)

    assert len(shadow_data["selected_sets"]) == 2  # n_models
    assert len(shadow_data["trained_results"]) == 2  # n_models
    for synthetic_data in shadow_data["trained_results"]:
        assert type(synthetic_data) is pd.DataFrame
        assert len(synthetic_data) == 5

    # Training sets should be disjoint
    assert set(shadow_data["selected_sets"][0]).isdisjoint(set(shadow_data["selected_sets"][1]))
    # Training sets should be unique
    assert len(shadow_data["selected_sets"][0]) == len(set(shadow_data["selected_sets"][0]))
    assert len(shadow_data["selected_sets"][1]) == len(set(shadow_data["selected_sets"][1]))


@pytest.mark.integration_test()
def test_train_and_fine_tune_tabddpm(cfg: DictConfig, tmp_path: Path) -> None:
    # Input
    train_set = pd.read_csv(
        "tests/unit/attacks/ensemble/assets/population_data/all_population.csv"
    )  # For testing purposes only.
    fine_tuning_set = copy.deepcopy(train_set)
    training_config_path = Path(cfg.shadow_training.training_json_config_paths.training_config_path)
    tmp_training_dir = tmp_path
    # We should move ``dataset_meta.json`` and ``trans_domain.json`` files to the ``tmp_training_dir``
    assert Path(cfg.shadow_training.training_json_config_paths.table_domain_file_path).exists()
    shutil.copyfile(
        cfg.shadow_training.training_json_config_paths.table_domain_file_path,
        tmp_training_dir / "trans_domain.json",
    )

    shutil.copyfile(
        cfg.shadow_training.training_json_config_paths.dataset_meta_file_path,
        tmp_training_dir / "dataset_meta.json",
    )
    with open(training_config_path, "r") as file:
        configs = ClavaDDPMTrainingConfig(**json.load(file))

    update_and_save_training_config(
        config=configs,
        data_dir=tmp_training_dir,
        final_config_json_path=tmp_training_dir / "trans.json",
        experiment_name="test_experiment",
        workspace_name="test_workspace",
    )

    model_runner = EnsembleAttackClavaDDPMModelRunner(cfg)
    model_runner.number_of_points_to_synthesize = 99
    model_runner.training_config.save_dir = tmp_training_dir

    train_result = model_runner.train_or_fine_tune_and_synthesize(train_set, synthesize=True)

    assert train_result.synthetic_data is not None
    assert type(train_result.synthetic_data) is pd.DataFrame
    assert len(train_result.synthetic_data) == 99

    assert train_result.models is not None
    assert type(train_result.models) is dict
    assert len(train_result.models) == 1  # Only one model (TabDDPM) is trained.

    # Now fine-tune the trained TabDDPM model on a small set of data
    fine_tuned_results = model_runner.train_or_fine_tune_and_synthesize(
        dataset=fine_tuning_set,
        synthesize=False,
        trained_model=train_result,
    )

    assert fine_tuned_results.synthetic_data is None
    assert fine_tuned_results.models is not None
    assert type(fine_tuned_results.models) is dict
    assert len(fine_tuned_results.models) == 1  # Only one model (TabDDPM) is fine-tuned.
