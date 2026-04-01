"""Provided test prediction probabilities of several attacked target models,
this script computes and saves the attack success metric.
"""

from logging import INFO
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from midst_toolkit.common.logger import log
from midst_toolkit.evaluation.privacy.mia_scoring import TprAtFpr


def load_target_challenge_labels_and_probabilities(
    metaclassifier_model_name: str, attack_results_path: Path, challenge_label_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads and returns the challenge labels and test prediction probabilities for
    a given target model.

    Args:
        metaclassifier_model_name: Name of the metaclassifier model used in the attack.
        attack_results_path: Path to the directory where attack results are saved.
        challenge_label_path: Path to the CSV file containing challenge labels.

    Return:
        A tuple containing:
            - test_target: Numpy array of true membership labels for the challenge points.
            - test_prediction_probabilities: Numpy array of prediction probabilities
              outputted by the metaclassifier for the challenge points.
    """
    attack_result_file_path = attack_results_path / f"{metaclassifier_model_name}_test_pred_proba.npy"
    assert attack_result_file_path.exists(), (
        f"No file found at {attack_result_file_path}. Make sure the path is correct, or run the attack on the target model first."
    )

    # Load the attack results containing test prediction probabilities.
    test_prediction_probabilities = np.load(attack_result_file_path)

    # Challenge labels are the true membership labels for the challenge points.
    test_target = pd.read_csv(challenge_label_path).to_numpy().squeeze()

    assert len(test_prediction_probabilities) == len(test_target), (
        "Number of challenge labels must match number of prediction probabilities."
    )

    return test_target, test_prediction_probabilities


def compute_attack_success_for_given_targets(
    target_model_config: DictConfig,
    target_ids: list[int],
    experiment_directory: Path,
    metaclassifier_model_name: str,
) -> None:
    """
    Computes and saves the attack success metric given the test prediction probabilities
    of several attacked target models by concatenating the target models' targets and predictions.
    NOTE: This function does not compute the average success across all models but rather
    treats all predictions and labels together for metric computation.

    Args:
        target_model_config: Configuration object for target models set in ``experiments_config.yaml``.
        target_ids: List of target model IDs to compute the attack success for.
        experiment_directory: Path to the base experiment directory where results are saved.
        metaclassifier_model_name: Name of the metaclassifier model used in the attack.
    """
    predictions = []
    targets = []
    individual_results = []
    for target_id in target_ids:
        # Override target model id in config as ``attack_probabilities_result_path`` and
        # ``challenge_label_path`` are dependent on it and change in runtime.
        target_model_config.target_model_id = target_id
        # Load challenge labels and prediction probabilities
        log(INFO, f"Loading challenge labels and prediction probabilities for target model ID {target_id}...")
        test_target, test_prediction_probabilities = load_target_challenge_labels_and_probabilities(
            metaclassifier_model_name=metaclassifier_model_name,
            attack_results_path=Path(target_model_config.attack_probabilities_result_path),
            challenge_label_path=Path(target_model_config.challenge_label_path),
        )
        predictions.append(test_prediction_probabilities)
        targets.append(test_target)
        # Also print the TPR@FPR=0.1 for each target model separately for reference.
        target_tpr_at_fpr = TprAtFpr.get_tpr_at_fpr(test_target, test_prediction_probabilities, fpr_threshold=0.1)
        log(INFO, f"Target model ID {target_id} has TPR of {target_tpr_at_fpr:.4f} at FPR={0.1}")
        individual_results.append(target_tpr_at_fpr)

    log(INFO, f"Individual TPR@FPR=0.1 results for each target model: {individual_results}")
    # Flatten arrays
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    assert len(predictions) == len(targets), "Number of predictions must match number of targets."

    # Compute TPR@FPR for all the target models
    tpr_at_fpr = TprAtFpr.get_tpr_at_fpr(targets, predictions, fpr_threshold=0.1)

    # Save the final attack success rate into a text file.
    metric_save_path = experiment_directory / f"attack_success_for_{metaclassifier_model_name}.txt"

    log(INFO, f"Saving attack success value of {tpr_at_fpr} TPR at FPR=0.1 to {metric_save_path}")
    with open(metric_save_path, "w") as f:
        f.write(f"Final TPR at FPR=0.1: {tpr_at_fpr:.4f}\n")


@hydra.main(config_path="configs", config_name="experiment_config_same_marginal_iid", version_base=None)
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
    compute_attack_success_for_given_targets(
        target_model_config=config.target_model,
        target_ids=target_ids,
        experiment_directory=Path(config.base_experiment_dir),
        metaclassifier_model_name=config.metaclassifier.meta_classifier_model_name,
    )


if __name__ == "__main__":
    main()
