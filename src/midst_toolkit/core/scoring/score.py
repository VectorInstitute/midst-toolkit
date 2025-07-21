"""Scoring program for MIA success evaluation.

Usage:
    score.py <input directory> <output directory>

Expected directory structure:
- <input directory>/ref/: Ground truth membership labels
- <input directory>/res/: Predictions for the same models

Structure inside each:
- <dataset>/<scenario>/<model_id>/solution.csv or prediction.csv
"""

import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .score_html import generate_html


FPR_THRESHOLD_LIST = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2]


# def tpr_at_fpr(true_membership: List, predictions: List, max_fpr = 0.1) -> float:
def tpr_at_fpr(true_membership: list[float], predictions: list[float], max_fpr: float = 0.1) -> float:
    """Calculates TPR at a given FPR threshold."""
    fpr, tpr, _ = roc_curve(true_membership, predictions)
    return max(tpr[fpr < max_fpr]) if np.any(fpr < max_fpr) else 0.0


def score(solutions: list, predictions: list) -> dict:
    """Calculates MIA scores based on true membership and predictions."""
    scores = {}
    for max_fpr in FPR_THRESHOLD_LIST:
        scores[f"TPR_FPR_{int(1e4 * max_fpr)}"] = tpr_at_fpr(solutions, predictions, max_fpr)
    fpr, tpr, _ = roc_curve(solutions, predictions)
    scores["fpr"] = fpr
    scores["tpr"] = tpr
    scores["AUC"] = roc_auc_score(solutions, predictions)
    scores["MIA"] = np.max(tpr - fpr)
    scores["accuracy"] = np.max(1 - (fpr + (1 - tpr)) / 2)
    return scores


if __name__ == "__main__":
    # assert len(os.sys.argv) == 3, "Usage: score.py <input directory> <output directory>"
    # input_dir = os.sys.argv[1]
    # output_dir = os.sys.argv[2]
    assert len(sys.argv) == 3, "Usage: score.py <input directory> <output directory>"
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    solutions_dir = os.path.join(input_dir, "ref")
    predictions_dir = os.path.join(input_dir, "res")

    dataset = os.listdir(solutions_dir)
    assert len(dataset) == 1, f"Expected one dataset in {solutions_dir}, got: {dataset}"
    # dataset = dataset[0]
    # print(f"[*] Scoring dataset: {dataset}")

    # solutions_dir = os.path.join(solutions_dir, dataset)
    # predictions_dir = os.path.join(predictions_dir, dataset)
    print(f"[*] Scoring dataset: {dataset[0]}")

    solutions_dir = os.path.join(solutions_dir, dataset[0])
    predictions_dir = os.path.join(predictions_dir, dataset[0])

    scenarios = sorted(os.listdir(solutions_dir))
    assert scenarios == sorted(os.listdir(predictions_dir)), "Mismatch in scenario folders"

    all_scores = {}

    for scenario in scenarios:
        print(f"[*] Processing scenario: {scenario}")
        scenario_solutions = []
        scenario_predictions = []

        solution_path = os.path.join(solutions_dir, scenario)
        prediction_path = os.path.join(predictions_dir, scenario)

        for model_id in os.listdir(solution_path):
            sol_file = os.path.join(solution_path, model_id, "solution.csv")
            pred_file = os.path.join(prediction_path, model_id, "prediction.csv")
            scenario_solutions.append(np.loadtxt(sol_file, delimiter=","))
            scenario_predictions.append(np.loadtxt(pred_file, delimiter=","))

        solutions = np.concatenate(scenario_solutions)
        predictions = np.concatenate(scenario_predictions)

        assert len(solutions) == len(predictions), "Mismatched lengths"
        assert np.all((predictions >= 0) & (predictions <= 1)), "Predictions not in [0,1]"

        # scores = score(solutions, predictions)
        scores = score(solutions.tolist(), predictions.tolist())
        all_scores[scenario] = scores
        print(f"[*] Scores for {scenario}: {scores}")

    # Write scores.txt
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "scores.txt"), "w") as f:
        for i, scenario in enumerate(scenarios):
            for key in ["AUC", "MIA", "accuracy"]:
                f.write(f"scenario{i + 1}_{key}: {all_scores[scenario][key]}\n")
            for max_fpr in FPR_THRESHOLD_LIST:
                key = f"TPR_FPR_{int(1e4 * max_fpr)}"
                f.write(f"scenario{i + 1}_{key}: {all_scores[scenario][key]}\n")

        avg = np.mean([all_scores[sc]["TPR_FPR_1000"] for sc in scenarios])
        f.write(f"average_TPR_FPR_1000: {avg}\n")

    # Generate scores.html
    html = generate_html(all_scores)
    with open(os.path.join(output_dir, "scores.html"), "w") as f:
        f.write(html)
