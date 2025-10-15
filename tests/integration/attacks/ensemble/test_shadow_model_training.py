import copy
import pickle
import shutil
from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from midst_toolkit.attacks.ensemble.data_utils import load_dataframe
from midst_toolkit.attacks.ensemble.rmia.shadow_model_training import (
    train_fine_tuned_shadow_models,
    train_shadow_on_half_challenge_data,
)
from midst_toolkit.attacks.ensemble.shadow_model_utils import (
    TrainingResult,
    fine_tune_tabddpm_and_synthesize,
    save_additional_tabddpm_config,
    train_tabddpm_and_synthesize,
)


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

    result_path = train_fine_tuned_shadow_models(
        n_models=2,
        n_reps=1,
        population_data=POPULATION_DATA,
        master_challenge_data=POPULATION_DATA[0:20],  # Limiting the data to 20 samples for faster test execution
        shadow_models_output_path=shadow_models_output_path,
        training_json_config_paths=cfg.shadow_training.training_json_config_paths,
        fine_tuning_config=cfg.shadow_training.fine_tuning_config,
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
    for result in shadow_data["fine_tuned_results"]:
        assert type(result) is TrainingResult
        assert result.synthetic_data is not None
        assert result.tables is not None
        assert result.models is not None
        assert result.configs is not None
        assert result.save_dir is not None
        assert result.relation_order is not None
        assert result.all_group_lengths_probabilities is not None
        assert type(result.synthetic_data) is pd.DataFrame

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
    result_path = train_shadow_on_half_challenge_data(
        n_models=2,
        n_reps=1,
        master_challenge_data=POPULATION_DATA[0:40],  # Limiting the data to 20 samples for faster test execution
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
    for result in shadow_data["trained_results"]:
        assert type(result) is TrainingResult
        assert result.synthetic_data is not None
        assert result.tables is not None
        assert result.models is not None
        assert result.configs is not None
        assert result.save_dir is not None
        assert result.relation_order is not None
        assert result.all_group_lengths_probabilities is not None
        assert type(result.synthetic_data) is pd.DataFrame

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
    tabddpm_config_path = Path(cfg.shadow_training.training_json_config_paths.tabddpm_training_config_path)
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
    configs, save_dir = save_additional_tabddpm_config(
        data_dir=tmp_training_dir,
        training_config_json_path=tabddpm_config_path,
        final_config_json_path=tmp_training_dir / "trans.json",
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
    assert train_result.synthetic_data is not None
    assert type(train_result.synthetic_data) is pd.DataFrame
    assert len(train_result.synthetic_data) == 99

    assert train_result.models is not None
    assert type(train_result.models) is dict
    assert len(train_result.models) == 1  # Only one model (TabDDPM) is trained.

    # Now fine-tune the trained TabDDPM model on a small set of data
    fine_tuned_results = fine_tune_tabddpm_and_synthesize(
        trained_models=train_result.models,
        fine_tune_set=fine_tuning_set,  # fine-tuning on the same data for testing purposes
        configs=configs,
        save_dir=save_dir,
        fine_tuning_diffusion_iterations=cfg.shadow_training.fine_tuning_config.fine_tune_diffusion_iterations,
        fine_tuning_classifier_iterations=cfg.shadow_training.fine_tuning_config.fine_tune_classifier_iterations,
        # Number of synthetic samples is defined according to tabddpm_training_config's classifier_scale value.
        synthesize=False,
    )
    assert fine_tuned_results.synthetic_data is None
    assert fine_tuned_results.models is not None
    assert type(fine_tuned_results.models) is dict
    assert len(fine_tuned_results.models) == 1  # Only one model (TabDDPM) is fine-tuned.
