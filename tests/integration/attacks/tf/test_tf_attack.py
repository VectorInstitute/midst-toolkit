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


# def test_tf_attack_whitebox_small_config():
#     # Set deterministic behavior
#     set_all_random_seeds(
#         seed=133742,
#         use_deterministic_torch_algos=True,
#         disable_torch_benchmarking=True,
#     )

#     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

#     phases = ["src_train", "train"]
#     base_path = Path("/projects/aieng/midst_competition/data/tabddpm")
#     base_path=Path('/projects/midst-experiments/tabddpm_midst_toolkit/train/')
#     config = {
#         "phases": phases,
#         "base_path": base_path,
#         "tabddpm_data_dir": base_path,
#         "target_model_subdir": "workspace/train_1/models",
#         "target_model_subdir": ".",
#         "model_type": "tabddpm",
#         "classifier_hidden_dim": 10,
#         "classifier_num_epochs": 20,
#         "samples_per_train_model": 30,
#         "sample_per_val_model": 10,
#         "num_noise_per_time_step": 10,
#         "timesteps_list": [5],
#         "addt_value_list": [0],
#         "predictions_file_format": "predictions_test_222",
#         "results_path": Path("/h/behnzaman/midst-toolkit/tests/integration/attacks/tf/test_tf_attack_results"),
#         "use_best_checkpoint": True,
#         "final_indices": [5],
#         "train_indices": [1, 2],
#         "val_indices": [3, 4],
#         "config_path": Path("/h/behnzaman/midst-toolkit/tests/integration/attacks/tf/data_configs/trans.json"),
#     }

#     MIA_performance_train, MIA_performance_test, MIA_performance_final = tf_attack(**config)
#     roc_auc_train, tpr_at_fpr_train = MIA_performance_train.values()
#     roc_auc_test, tpr_at_fpr_test = MIA_performance_test.values()
#     roc_auc, tpr_at_fpr = MIA_performance_final.values()




#     assert roc_auc == pytest.approx(0.56765, abs=1e-8)
#     assert tpr_at_fpr == pytest.approx(0.08, abs=1e-8)
    
#     assert roc_auc_train == pytest.approx(0.56555, abs=1e-8)
#     assert tpr_at_fpr_train == pytest.approx(0.075, abs=1e-8)
    
#     assert roc_auc_test == pytest.approx(0.5548500000000001, abs=1e-8)
#     assert tpr_at_fpr_test == pytest.approx(0.06, abs=1e-8)
    
#     unset_all_random_seeds()
#     os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def test_tf_attack_whitebox_small_config_new_setup():
    # Set deterministic behavior
    set_all_random_seeds(
        seed=133742,
        use_deterministic_torch_algos=True,
        disable_torch_benchmarking=True,
    )

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    phases = ["src_train", "train"]
    base_path=Path('/projects/midst-experiments/tabddpm_midst_toolkit/train/')
    config = {
        "phases": phases,
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
        "predictions_file_format": "predictions_test_222",
        "results_path": Path("/h/behnzaman/midst-toolkit/tests/integration/attacks/tf/test_tf_attack_results"),
        "use_best_checkpoint": True,
        "final_indices": [5],
        "train_indices": [1, 2],
        "val_indices": [3, 4],
        "config_path": Path("/h/behnzaman/midst-toolkit/tests/integration/attacks/tf/data_configs/trans.json"),
    }

    MIA_performance_train, MIA_performance_test, MIA_performance_final = tf_attack(**config)
    roc_auc_train, tpr_at_fpr_train = MIA_performance_train.values()
    roc_auc_test, tpr_at_fpr_test = MIA_performance_test.values()
    roc_auc, tpr_at_fpr = MIA_performance_final.values()




    assert roc_auc == pytest.approx(0.5179, abs=1e-8)
    assert tpr_at_fpr == pytest.approx(0.07, abs=1e-8)
    
    assert roc_auc_train == pytest.approx(0.48106250000000006, abs=1e-8)
    assert tpr_at_fpr_train == pytest.approx(0.12, abs=1e-8)
    
    assert roc_auc_test == pytest.approx(0.479625, abs=1e-8)
    assert tpr_at_fpr_test == pytest.approx(0.1, abs=1e-8)
    
    unset_all_random_seeds()
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

if __name__ == "__main__":
    test_tf_attack_whitebox_small_config_new_setup()
