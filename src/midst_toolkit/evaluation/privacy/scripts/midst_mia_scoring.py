import argparse
import os
from logging import INFO
from pathlib import Path

import numpy as np

from midst_toolkit.common.logger import log
from midst_toolkit.evaluation.privacy.mia_metrics import DEFAULT_FPR_THRESHOLDS, MembershipInferenceMetrics
from midst_toolkit.evaluation.privacy.mia_utils import generate_html


def compute_mia_scores_across_scenarios(
    labels_scenarios_dir: Path, preds_scenarios_dir: Path, fpr_thresholds: list[float]
) -> dict[str, dict[str, float | np.ndarray]]:
    """
    Give a path to a set of scenario folders holding membership predictions and labels, this function runs scoring
    for all of the scenarios and stores the results in a dictionary keyed by the scenario folder name. The values
    of the dictionary are the sets of scores for the membership predictions keyed by the name of the metric. The
    folder structure at each of the paths is assumed to be <scenario>/<model_id>/solutions.csv or predictions.csv.

    Args:
        labels_scenarios_dir: Directory containing scenario membership labels.
        preds_scenarios_dir: Directory containing scenario membership predictions.
        fpr_thresholds: false positive rate thresholds to measure TRP @ FPR metrics for the membership inference. For
            more details, see documentation in ``compute_mia_metrics`` and the sub-methods therein.

    Returns:
        Scoring for the various scenarios folders in the provided directories. The dictionary is keyed by the
        scenario folder name. The values of the dictionary are the sets of scores for the membership predictions
        keyed by the name of the metric.
    """
    scenarios = set(os.listdir(labels_scenarios_dir))
    assert scenarios == set(os.listdir(preds_scenarios_dir)), "Label and Predictions scenario folders do not match."

    log(INFO, f"Processing Scenarios: {scenarios}")

    scores_by_scenario: dict[str, dict[str, float | np.ndarray]] = {}
    for scenario in scenarios:
        log(INFO, f"Processing scenario: {scenario}")

        # By default, computing all metrics
        metrics = MembershipInferenceMetrics(fpr_thresholds=fpr_thresholds)

        scenario_labels = []
        scenario_predictions = []

        label_scenario_path = labels_scenarios_dir / scenario
        prediction_scenario_path = preds_scenarios_dir / scenario

        # NOTE: Performance is scored ACROSS model IDs within a scenario. That is, for each scenario, if you have
        # multiple model ID folders, the predictions and labels are agglomerated and treated as ONE set of membership
        # predictions for metrics computation
        for model_id in os.listdir(label_scenario_path):
            log(INFO, f"Processing Model ID: {model_id}")

            label_file = label_scenario_path / model_id / "solution.csv"
            prediction_file = prediction_scenario_path / model_id / "prediction.csv"

            scenario_labels.append(np.loadtxt(label_file, delimiter=","))
            scenario_predictions.append(np.loadtxt(prediction_file, delimiter=","))

        # Grouping all predictions and labels ACROSS the model IDs for metric computation.
        all_labels = np.concatenate(scenario_labels)
        all_predictions = np.concatenate(scenario_predictions)

        metrics.compute(all_labels, all_predictions)

        scores = metrics.to_dict()
        scores_by_scenario[scenario] = scores
        log(INFO, f"Scores for {scenario}: {scores}")

    return scores_by_scenario


if __name__ == "__main__":
    """
    NOTE: For this script, performance is scored ACROSS model IDs within a scenario. That is, for each scenario, if
    you have  multiple model ID folders (see structure described below), the predictions and labels are agglomerated
    and treated as ONE set of membership predictions for metrics computation. This matches the implementation in the
    MIDST repository: https://github.com/VectorInstitute/MIDST/blob/main/codabench_bundles/midst_blackbox_multi_table/scoring_programs/final_scoring_program/scoring.py
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preds_labels_directory",
        required=True,
        type=str,
        help=(
            "The path to the directory containing both the membership labels files and predictions files. These "
            "should be in CSV format. The file structure of the directory is assumed to follow the structure:\n"
            " - <preds_labels_directory>/ref/: Ground truth membership labels\n - <preds_labels_directory>/res/: "
            "Predictions for the same models\n\nStructure inside each:\n - <dataset>/<scenario>/<model_id>"
            "/solution.csv or prediction.csv.\n\nNote that there should only be a single dataset folder, but there "
            "may be multiple scenario folders through which this script will iterate. The same scenario folder must "
            "be present in both the labels and predictions directories."
        ),
    )
    parser.add_argument(
        "--output_directory",
        required=True,
        type=str,
        help=(
            "This is the directory to which the results of the evaluations will be written. One will be created at "
            "the specified path if it does not already exists."
        ),
    )

    parser.add_argument(
        "--fpr_thresholds",
        required=False,
        type=float,
        nargs="+",
        help=(
            "This is a list of floats to specify the various FPR thresholds to compute the TPR at FPR metrics for "
            "the membership inference scoring. If unspecified, it defaults to DEFAULT_FPR_THRESHOLDS"
        ),
    )

    args = parser.parse_args()

    preds_labels_directory = Path(args.preds_labels_directory)
    output_directory = Path(args.output_directory)
    fpr_thresholds: list[float] = args.fpr_thresholds if args.fpr_thresholds else DEFAULT_FPR_THRESHOLDS

    labels_directory = preds_labels_directory / "ref"
    predictions_directory = preds_labels_directory / "res"

    labels_datasets = os.listdir(labels_directory)
    preds_datasets = os.listdir(predictions_directory)
    # We assume that there is only one datasets folder for the evaluation script
    assert len(labels_datasets) == 1, f"Expected one dataset in {labels_directory}, got: {labels_datasets}"
    assert len(preds_datasets) == 1, f"Expected one dataset in {preds_datasets}, got: {preds_datasets}"

    labels_dataset = labels_datasets[0]
    preds_dataset = preds_datasets[0]

    assert labels_dataset == preds_dataset, "Dataset names are different for labels and preds directories."
    log(INFO, f"Scoring Dataset: {labels_dataset}")

    # These are the "scenarios" or collections of predictions to be analyzed.
    labels_scenarios_dir = labels_directory / labels_dataset
    preds_scenarios_dir = predictions_directory / preds_dataset

    scores_by_scenario = compute_mia_scores_across_scenarios(labels_scenarios_dir, preds_scenarios_dir, fpr_thresholds)

    if output_directory.exists():
        log(INFO, f"Output Directory: {output_directory} already exists. Writing to existing directory")
    else:
        os.makedirs(output_directory)

    with open(output_directory / "scores.txt", "w") as output_file:
        for scenario_key, metrics in scores_by_scenario.items():
            for metric_key in ("auc", "mia", "balanced_accuracy"):
                output_file.write(f"{scenario_key}_{metric_key}: {metrics[metric_key]}\n")

            for max_fpr in fpr_thresholds:
                metric_key = f"TPR_FPR_{int(1e4 * max_fpr)}"
                output_file.write(f"{scenario_key}_{metric_key}: {metrics[metric_key]}\n")

        # Compute the mean TPR @ FPR = 0.1 across all scenarios
        tpr_at_fpr_0_1_collection: list[float] = []
        for metrics in scores_by_scenario.values():
            if "TPR_FPR_1000" in metrics:
                tpr_at_fpr_0_1 = metrics["TPR_FPR_1000"]
                assert isinstance(tpr_at_fpr_0_1, float)
                tpr_at_fpr_0_1_collection.append(tpr_at_fpr_0_1)

        average_tpr_at_fpr_0_1 = np.mean(tpr_at_fpr_0_1_collection)
        output_file.write(f"average_TPR_FPR_1000: {average_tpr_at_fpr_0_1}\n")

    # Generate scores.html
    html = generate_html(scores_by_scenario)
    with open(output_directory / "scores.html", "w") as output_file:
        output_file.write(html)
