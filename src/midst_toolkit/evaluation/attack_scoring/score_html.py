import io

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


# To avoid launching a python subprocess
mpl.use("Agg")

DEFAULT_COLUMN_RENAME = {
    "balanced_accuracy": "Balanced Accuracy",
    "auc": "AUC-ROC",
    "mia": "MIA Score",
    "fpr": "fpr",
    "tpr": "tpr",
    "TPR_FPR_10": "TPR @ 0.001 FPR",
    "TPR_FPR_100": "TPR @ 0.01 FPR",
    "TPR_FPR_500": "TPR @ 0.05 FPR",
    "TPR_FPR_1000": "TPR @ 0.1 FPR",
    "TPR_FPR_1500": "TPR @ 0.15 FPR",
    "TPR_FPR_2000": "TPR @ 0.2 FPR",
}


def _image_to_html(figure: Figure) -> str:
    """
    Converts a matplotlib plot to SVG.

    Args:
        figure: Matplotlib figure object to be converted to SVG.

    Returns:
        SVG figure flattened as a string.
    """
    iostring = io.StringIO()
    figure.savefig(iostring, format="svg", bbox_inches=0, dpi=300)
    iostring.seek(0)

    return iostring.read()


def _generate_roc_figure(false_positive_rates: np.ndarray, true_positive_rates: np.ndarray) -> Figure:
    """
    Generates a ROC plot from the computed false positive rates (FPRs) and true positive rates (TPRs) at
    various (and corresponding) thresholds. The returned value is a matplotlib figure. The figure is composed of two
    subfigures. The first is a standard ROC plot. The second is a log-log plot of the ROC for a different
    perspective on the metrics.

    Args:
        false_positive_rates: False positive rates associated with some set of predictions at various thresholds.
        true_positive_rates: True positive rates associated with some set of predictions at various thresholds.

    Returns:
        A matplotlib Figure plotting ROC using the FPR and TPR values at various classification thresholds.
    """
    figure, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 3.5))

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.plot([0, 1], [0, 1], ls=":", color="grey")

    ax2.semilogx()
    ax2.semilogy()
    ax2.set_xlim(1e-5, 1)
    ax2.set_ylim(1e-5, 1)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.plot([0, 1], [0, 1], ls=":", color="grey")

    ax1.plot(false_positive_rates, true_positive_rates)
    ax2.plot(false_positive_rates, true_positive_rates)

    return figure


def _generate_table(
    scenario_scores: dict[str, dict[str, float | np.ndarray]], column_name_replacement: dict[str, str]
) -> pd.DataFrame:
    """
    Generates a DataFrame with the scores from the membership scoring function. Column names are replaced with the
    contents of ``column_name_replacement``. The keys of ``scores`` and ``column_name_replacement`` must be the
    same. Each row represents a separate set of scores for a specific scenario.

    Args:
        scenario_scores: Scores for a set of scenarios. The outer dictionary is keyed by a scenario name and each
            inner  dictionary contains metrics for membership inference from the specific scenario. The keys of the
            inner dictionaries are the names of the metrics and values are the metric scores.
        column_name_replacement: How the column names should be renamed in the dataframe. The keys of this dictionary
            need to match those of scores in the inner dictionaries.

    Returns:
        A dataframe of the scores results with some optional column name replacements applied.
    """
    table_scores: dict[str, list[float]] = {}
    scenario_names = []
    for scenario, scores in scenario_scores.items():
        assert set(scores.keys()) == set(column_name_replacement.keys()), (
            "Keys of scores and column_name_replacement do not match"
        )
        scenario_names.append(scenario)

        for metric_name, metric_value in scores.items():
            if metric_name not in {"fpr", "tpr"}:
                assert isinstance(metric_value, float)
                if metric_name in table_scores:
                    table_scores[metric_name].append(metric_value)
                # First entry for the metric
                else:
                    table_scores[metric_name] = [metric_value]

    table = pd.DataFrame.from_dict(table_scores)
    table.columns = [column_name_replacement[c] for c in table.columns]
    # Add scenario names column at the front.
    table.insert(0, "Scenario Names", scenario_names)
    return table


def generate_html(
    scenario_scores: dict[str, dict[str, float | np.ndarray]], column_name_replacement: dict[str, str] | None = None
) -> str:
    """
    Generates the HTML document as a string, containing the various detailed scores.

    Args:
        scenario_scores: Scores for a set of scenarios. The outer dictionary is keyed by a scenario name and each
            inner  dictionary contains metrics for membership inference from the specific scenario. The keys of the
            inner dictionaries are the names of the metrics and values are the metric scores.
        column_name_replacement: How the column names should be renamed in the dataframe. If None, it defaults to the
            renaming convention based on the default metrics in the scoring function. The keys of this dictionary need
            to match those of scores. Defaults to None.

    Returns:
        An HTML document as a string which renders the scores in a more visual way.
    """
    column_name_replacement = column_name_replacement if column_name_replacement else DEFAULT_COLUMN_RENAME

    imgs_html = ""
    for scenario in scenario_scores:
        fpr = scenario_scores[scenario]["fpr"]
        tpr = scenario_scores[scenario]["tpr"]
        assert isinstance(fpr, np.ndarray)
        assert isinstance(tpr, np.ndarray)
        fig = _generate_roc_figure(fpr, tpr)
        fig.tight_layout(pad=1.0)

        imgs_html = f"{imgs_html}<h2>{scenario}</h2><div>{_image_to_html(fig)}</div>"

    table = _generate_table(scenario_scores, column_name_replacement)
    table_html = table.to_html(border=0, float_format="{:0.4f}".format, escape=False)

    with open("src/midst_toolkit/evaluation/attack_scoring/html_elements.txt", "r") as html_file:
        html_string = html_file.read()
        html_string = html_string.replace("MIA_TABLES", table_html)
        return html_string.replace("MIA_SCENARIOS", imgs_html)
