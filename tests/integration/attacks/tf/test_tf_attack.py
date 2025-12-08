import os
from pathlib import Path

import pytest

from midst_toolkit.attacks.tf.tf_attack import tf_attack
from midst_toolkit.common.random import (
    set_all_random_seeds,
    unset_all_random_seeds,
)


def test_tf_attack_whitebox_tiny_config_midst_toolkit():
    set_all_random_seeds(
        seed=133742,
        use_deterministic_torch_algos=True,
        disable_torch_benchmarking=True,
    )

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    base_path = Path(os.path.dirname(__file__)) / "assets" / "tabddpm_models"
    config = {
        "base_path": base_path,
        "tabddpm_data_dir": base_path,
        "target_model_subdir": ".",
        "model_type": "tabddpm",
        "classifier_hidden_dim": 20,
        "classifier_num_epochs": 50,
        "samples_per_train_model": 3000,
        "sample_per_val_model": 10,
        "num_noise_per_time_step": 30,
        "timesteps_list": [5, 10],
        "addt_value_list": [0],
        "predictions_file_format": "predictions_test_222",
        "results_path": Path(__file__).parent / "test_tf_attack_results",
        "test_indices": [5, 6],
        "train_indices": [1, 2],
        "val_indices": [3, 4],
        "meta_dir": Path(__file__).parent / "data_configs",
        "classifier_learning_rate": 1e-4,
    }

    mia_performance_train, mia_performance_val, mia_performance_test = tf_attack(**config)
    roc_auc_train = mia_performance_train["roc_auc"]
    tpr_at_fpr_train = mia_performance_train["max_tpr"]
    roc_auc_val = mia_performance_val["roc_auc"]
    tpr_at_fpr_val = mia_performance_val["max_tpr"]
    roc_auc_test = mia_performance_test["roc_auc"]
    tpr_at_fpr_test = mia_performance_test["max_tpr"]

    assert roc_auc_train == pytest.approx(0.6659875, abs=1e-8)
    assert tpr_at_fpr_train == pytest.approx(0.265, abs=1e-8)

    assert roc_auc_val == pytest.approx(0.6328874999999999, abs=1e-8)
    assert tpr_at_fpr_val == pytest.approx(0.23, abs=1e-8)

    assert roc_auc_test == pytest.approx(0.6519875, abs=1e-8)
    assert tpr_at_fpr_test == pytest.approx(0.235, abs=1e-8)

    unset_all_random_seeds()
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


if __name__ == "__main__":
    test_tf_attack_whitebox_tiny_config_midst_toolkit()
