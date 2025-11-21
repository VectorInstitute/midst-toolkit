"""Provided test prediction probabilities of several attacked target models,
this script computes and saves the attack success metric.
"""

from logging import INFO
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from midst_toolkit.attacks.ensemble.metric_utils import get_tpr_at_fpr
from midst_toolkit.common.logger import log


def load_target_challenge_labels_and_probabilities(
    config: DictConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads and returns the challenge labels and test prediction probabilities for
    a given target model.

    Args:
        config: Configuration object set in ``experiments_config.yaml``.
    """
    metaclassifier_model_name = config.metaclassifier.meta_classifier_model_name
    #  ``attack_probabilities_result_path`` is based on the target model's id.
    attack_results_path = Path(config.target_model.attack_probabilities_result_path)
    attack_result_file_path = attack_results_path / f"{metaclassifier_model_name}_test_pred_proba.npy"
    assert attack_result_file_path.exists(), (
        f"No file found at {attack_result_file_path}. Make sure the path is correct, or run the attack on the target model first."
    )

    # Load the attack results containing test prediction probabilities.
    test_prediction_probabilities = np.load(attack_result_file_path)

    # Challenge labels are the true membership labels for the challenge points.
    challenge_label_path = Path(config.target_model.challenge_label_path)
    test_target = pd.read_csv(challenge_label_path).to_numpy().squeeze()

    assert len(test_prediction_probabilities) == len(test_target), (
        "Number of challenge labels must match number of prediction probabilities."
    )

    return test_target, test_prediction_probabilities


def compute_attack_success_for_given_targets(config: DictConfig, target_ids: list[int]) -> None:
    """
    Computes and saves the attack success metric given the test prediction probabilities
    of several attacked target models by concatenating the target models' targets and predictions.

    Args:
        config: Configuration object set in ``experiments_config.yaml``.
        target_ids: List of target model IDs to compute the attack success for.
    """
    experiment_directory = Path(config.base_experiment_dir)
    predictions = []
    targets = []
    for target_id in target_ids:
        # Override target model id in config
        config.target_model.target_model_id = target_id
        # Load challenge labels and prediction probabilities
        test_target, test_prediction_probabilities = load_target_challenge_labels_and_probabilities(config)
        predictions.append(test_prediction_probabilities)
        targets.append(test_target)

    # Flatten arrays
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    assert len(predictions) == len(targets), "Number of predictions must match number of targets."

    # Compute TPR@FPR for all the target models
    tpr_at_fpr = get_tpr_at_fpr(targets, predictions, max_fpr=0.1)

    # Save the final attack success rate into a text file.
    metaclassifier_model_name = config.metaclassifier.meta_classifier_model_name
    metric_save_path = experiment_directory / f"attack_success_for_{metaclassifier_model_name}.txt"

    log(INFO, f"Saving attack success value of {tpr_at_fpr} TPR at FPR=0.1 to {metric_save_path}")
    with open(metric_save_path, "w") as f:
        f.write(f"Final TPR at FPR=0.1: {tpr_at_fpr:.4f}\n")


@hydra.main(config_path="configs", config_name="experiment_config", version_base=None)
def main(
    config: DictConfig,
) -> None:
    """
    Main function to compute and save the attack success metric given the test prediction probabilities
    of several attacked target models.

    Args:
        config: Configuration object set in ``experiments_config.yaml``.
    """
    assert config.attack_success_computation.target_ids_to_test is not None, (
        "Please specify target model IDs to compute attack success for in the config "
        "by specifying `attack_success_computation.target_ids_to_test`."
    )
    target_ids = list(config.attack_success_computation.target_ids_to_test)
    log(INFO, f"Computing attack success for target model IDs: {target_ids}...")
    compute_attack_success_for_given_targets(config, target_ids)


if __name__ == "__main__":
    main()
