import os
import sys
from pathlib import Path

import pytest


# Add paths
sys.path.append("/h/behnzaman/")
sys.path.insert(0, "/h/behnzaman/midst-experiments/deps/TF_attack/")

from midst_toolkit.attacks.tf.tf_attack import run_experiment
from midst_toolkit.common.random import (
    set_all_random_seeds,
    unset_all_random_seeds,
)


def test_tf_attack_whitebox_small_config():
    # Set deterministic behavior
    set_all_random_seeds(
        seed=133742,
        use_deterministic_torch_algos=True,
        disable_torch_benchmarking=True,
    )

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    phases = ["src_train", "train"]
    base_path = Path("/projects/aieng/midst_competition/data/tabddpm/")

    config = {
        "phases": phases,
        "base_path": base_path,
        "tabddpm_data_dir": base_path,
        "n_synthetic_data_points": 42,
        "new_model": "workspace/train_1/models",
        "model_type": "tabddpm",
        "hidden_dim": 10,
        "num_epochs": 20,
        "data_per_model": 30,
        "test_data_model": 10,
        "noise_num": 10,
        "n_trained_models_list": [10],
        "test_model_num": 10,
        "t_value_list": [5],
        "addt_value_list": [0],
        "predictions_file_format": "predictions_test_222",
        "results_path": "/h/behnzaman/midst-toolkit/tests/integration/attacks/tf/test_tf_attack_results",
        "use_best_checkpoint": True,
        "final_indices": [5, 6],
        "train_indices": [1, 2],
        "test_indices": [3, 4],
    }

    roc_auc, tpr_at_fpr = run_experiment(**config)

    unset_all_random_seeds()
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    assert roc_auc == pytest.approx(0.567475, abs=1e-8)
    assert tpr_at_fpr == pytest.approx(0.08, abs=1e-8)
