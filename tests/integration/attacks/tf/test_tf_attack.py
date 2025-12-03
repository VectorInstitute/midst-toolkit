# mypy: disable-error-code=no-untyped-def,has-type
import os
import sys
from pathlib import Path

import pytest


# Add paths
sys.path.append("/h/behnzaman/")
sys.path.insert(0, "/h/behnzaman/midst-experiments/deps/TF_attack/")

from midst_toolkit.attacks.tf.tf_attack import tf_attack
from midst_toolkit.common.random import (
    set_all_random_seeds,
    unset_all_random_seeds,
)


def test_tf_attack_whitebox_small_config_new_setup():
    # Set deterministic behavior
    set_all_random_seeds(
        seed=133742,
        use_deterministic_torch_algos=True,
        disable_torch_benchmarking=True,
    )

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    base_path = Path("tests/integration/attacks/tf/assets/tabddpm_models")
    config = {
        "base_path": base_path,
        "tabddpm_data_dir": base_path,
        "target_model_subdir": ".",
        "model_type": "tabddpm",
        "classifier_hidden_dim": 10,
        "classifier_num_epochs": 20,
        "samples_per_train_model": 300,
        "sample_per_val_model": 100,
        "num_noise_per_time_step": 10,
        "timesteps_list": [5],
        "addt_value_list": [0],
        "predictions_file_format": "predictions_test_2",
        "results_path": Path("tests/integration/attacks/tf/results"),
        "use_best_checkpoint": True,
        "test_indices": [5],
        "train_indices": [1, 2],
        "val_indices": [3, 4],
    }

    mia_performance_train, mia_performance_val, mia_performance_test = tf_attack(**config)
    tpr_at_fpr_train, roc_auc_train = mia_performance_train.values()
    tpr_at_fpr_val, roc_auc_val = mia_performance_val.values()
    tpr_at_fpr_test, roc_auc_test = mia_performance_test.values()

    assert roc_auc_train == pytest.approx(0.48196249999999996, abs=1e-8)
    assert tpr_at_fpr_train == pytest.approx(0.125, abs=1e-8)

    assert roc_auc_val == pytest.approx(0.4794125000000001, abs=1e-8)
    assert tpr_at_fpr_val == pytest.approx(0.095, abs=1e-8)

    assert roc_auc_test == pytest.approx(0.5185000000000001, abs=1e-8)
    assert tpr_at_fpr_test == pytest.approx(0.1, abs=1e-8)

    unset_all_random_seeds()
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


if __name__ == "__main__":
    test_tf_attack_whitebox_small_config_new_setup()
