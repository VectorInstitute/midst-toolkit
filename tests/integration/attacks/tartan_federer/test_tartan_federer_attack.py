import os
from pathlib import Path

import pytest

from midst_toolkit.attacks.tartan_federer.tartan_federer_attack import tartan_federer_attack
from midst_toolkit.common.random import (
    set_all_random_seeds,
    unset_all_random_seeds,
)


@pytest.mark.integration_test()
def test_tf_attack_whitebox_tiny_config_midst_toolkit():
    set_all_random_seeds(
        seed=133742,
        use_deterministic_torch_algos=True,
        disable_torch_benchmarking=True,
    )

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    base_path = Path(os.path.dirname(__file__)) / "assets" / "tabddpm_models"
    config = {
        "population_data_dir": Path(__file__).parent / "assets" / "population_data",
        "model_data_dir": base_path,
        "target_model_subdir": Path("."),
        "model_type": "tabddpm",
        "classifier_hidden_dim": 20,
        "classifier_num_epochs": 200,
        "samples_per_train_model": 3000,
        "sample_per_val_model": 10,
        "num_noise_per_time_step": 30,
        "timesteps": [5, 10, 15],
        "additional_timesteps": [0],
        "predictions_file_format": "challenge_label_predictions",
        # TODO: Make results path a temp directory
        "results_path": Path(__file__).parent / "assets" / "tartan_federer_attack_results",
        "test_indices": [5, 6],
        "train_indices": [1, 2],
        "val_indices": [3, 4],
        "columns_for_deduplication": ["trans_id", "balance"],
        # TODO: Make results path a temp directory
        "meta_dir": Path(__file__).parent / "assets" / "data_configs",
        "classifier_learning_rate": 1e-4,
    }

    mia_performance_train, mia_performance_val, mia_performance_test = tartan_federer_attack(**config)
    roc_auc_train = mia_performance_train["roc_auc"]
    tpr_at_fpr_train = mia_performance_train["max_tpr"]
    roc_auc_val = mia_performance_val["roc_auc"]
    tpr_at_fpr_val = mia_performance_val["max_tpr"]
    roc_auc_test = mia_performance_test["roc_auc"]
    tpr_at_fpr_test = mia_performance_test["max_tpr"]

    assert roc_auc_train == pytest.approx(0.4469875, abs=1e-8)
    assert tpr_at_fpr_train == pytest.approx(0.08, abs=1e-8)

    assert roc_auc_val == pytest.approx(0.5054624999999999, abs=1e-8)
    assert tpr_at_fpr_val == pytest.approx(0.125, abs=1e-8)

    assert roc_auc_test == pytest.approx(0.4937875, abs=1e-8)
    assert tpr_at_fpr_test == pytest.approx(0.115, abs=1e-8)

    unset_all_random_seeds()
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


@pytest.mark.integration_test()
def test_tf_attack_whitebox_tiny_config_midst_toolkit_single_model():
    set_all_random_seeds(
        seed=133742,
        use_deterministic_torch_algos=True,
        disable_torch_benchmarking=True,
    )

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    base_path = Path(os.path.dirname(__file__)) / "assets" / "tabddpm_models"
    config = {
        "population_data_dir": Path(__file__).parent / "assets" / "population_data",
        "model_data_dir": base_path,
        "target_model_subdir": Path("."),
        "model_type": "tabddpm",
        "classifier_hidden_dim": 100,
        "classifier_num_epochs": 200,
        "samples_per_train_model": 3000,
        "sample_per_val_model": 10,
        "num_noise_per_time_step": 30,
        "timesteps": [5, 10, 15],
        "additional_timesteps": [0],
        "predictions_file_format": "challenge_label_predictions",
        # TODO: Make results path a temp directory
        "results_path": Path(__file__).parent / "assets" / "tartan_federer_attack_results",
        "test_indices": [3],
        "train_indices": [1],
        "val_indices": [2],
        "columns_for_deduplication": ["trans_id", "balance"],
        # TODO: Make results path a temp directory
        "meta_dir": Path(__file__).parent / "assets" / "data_configs",
        "classifier_learning_rate": 1e-4,
    }

    mia_performance_train, mia_performance_val, mia_performance_test = tartan_federer_attack(**config)
    roc_auc_train = mia_performance_train["roc_auc"]
    tpr_at_fpr_train = mia_performance_train["max_tpr"]
    roc_auc_val = mia_performance_val["roc_auc"]
    tpr_at_fpr_val = mia_performance_val["max_tpr"]
    roc_auc_test = mia_performance_test["roc_auc"]
    tpr_at_fpr_test = mia_performance_test["max_tpr"]

    assert roc_auc_train == pytest.approx(0.5046999999999999, abs=1e-8)
    assert tpr_at_fpr_train == pytest.approx(0.09, abs=1e-8)

    assert roc_auc_val == pytest.approx(0.47159999999999996, abs=1e-8)
    assert tpr_at_fpr_val == pytest.approx(0.12, abs=1e-8)

    assert roc_auc_test == pytest.approx(0.46390000000000003, abs=1e-8)
    assert tpr_at_fpr_test == pytest.approx(0.16, abs=1e-8)

    unset_all_random_seeds()
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


@pytest.mark.integration_test()
def test_tf_attack_whitebox_tiny_config_midst_toolkit_no_validation():
    set_all_random_seeds(
        seed=133742,
        use_deterministic_torch_algos=True,
        disable_torch_benchmarking=True,
    )

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    base_path = Path(os.path.dirname(__file__)) / "assets" / "tabddpm_models"
    config = {
        "population_data_dir": Path(__file__).parent / "assets" / "population_data",
        "model_data_dir": base_path,
        "target_model_subdir": Path("."),
        "model_type": "tabddpm",
        "classifier_hidden_dim": 100,
        "classifier_num_epochs": 200,
        "samples_per_train_model": 3000,
        "sample_per_val_model": 10,
        "num_noise_per_time_step": 30,
        "timesteps": [5, 10, 15],
        "additional_timesteps": [0],
        "predictions_file_format": "challenge_label_predictions",
        # TODO: Make results path a temp directory
        "results_path": Path(__file__).parent / "assets" / "tartan_federer_attack_results",
        "test_indices": [2],
        "train_indices": [1],
        "val_indices": None,
        "columns_for_deduplication": ["trans_id", "balance"],
        # TODO: Make results path a temp directory
        "meta_dir": Path(__file__).parent / "assets" / "data_configs",
        "classifier_learning_rate": 1e-4,
    }

    mia_performance_train, mia_performance_val, mia_performance_test = tartan_federer_attack(**config)
    roc_auc_train = mia_performance_train["roc_auc"]
    tpr_at_fpr_train = mia_performance_train["max_tpr"]
    roc_auc_test = mia_performance_test["roc_auc"]
    tpr_at_fpr_test = mia_performance_test["max_tpr"]

    assert mia_performance_val is None

    assert roc_auc_train == pytest.approx(0.4996999999999999, abs=1e-8)
    assert tpr_at_fpr_train == pytest.approx(0.07, abs=1e-8)

    assert roc_auc_test == pytest.approx(0.5174, abs=1e-8)
    assert tpr_at_fpr_test == pytest.approx(0.13, abs=1e-8)

    unset_all_random_seeds()
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
