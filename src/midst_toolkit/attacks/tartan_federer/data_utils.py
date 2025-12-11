from __future__ import annotations

import io
import json
import os
import pickle
from logging import INFO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from midst_toolkit.common.logger import log
from midst_toolkit.evaluation.privacy.mia_scoring import TprAtFpr
from midst_toolkit.models.clavaddpm.data_loaders import Table, Tables, get_info_from_domain
from midst_toolkit.models.clavaddpm.enumerations import RelationOrder


ArrayDict = dict[str, np.ndarray]
TensorDict = dict[str, torch.Tensor]


class CustomUnpickler(pickle.Unpickler):
    """Extending the Unpickler class to include the find_class function below."""

    def find_class(self, module: str, name: str) -> Any:
        """
        Overriding the super class find_class function to return a function that takes in bytes and uses torch.load
        to load the object and map it to a cpu.

        Args:
            module: The kind of module to be loaded.
            name: How to do the loading of the module.

        Returns:
            In the specific case of model == "torch.storage" and name == "_load_from_bytes" return a function that
            takes in bytes and uses torch.load to load the object and map it to a cpu. Otherwise uses the default
            functionality for the Unpickler.
        """
        # Force CUDA tensors to load on CPU
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def load_multi_table_customized(
    data_dir: Path, meta_dir: Path | None = None, train_name: str = "train.csv"
) -> tuple[Tables, RelationOrder, dict[str, Any]]:
    """
    Implements a custom loading function for the multi-table setting. This functionality is similar to that of
    load_tables in ``midst_toolkit.models.clavaddpm.data_loaders`` but with extra filtration steps.

    The meta data for the table information should be named 'dataset_meta.json' Each table should have a domain file
    called {table_name}_domain.json.

    Args:
        data_dir: The directory to load the dataset from.
        meta_dir: An optional separate path containing the meta data information about the tables and datasets.
            If None, this function looks for 'dataset_meta.json' in the ``data_dir`` path. Defaults to None.
        train_name: Name of the file containing the table data. This should exist in the ``data_dir`` path.
            Defaults to "train.csv".

    Raises:
        ValueError: Throws a value error if any of the columns in any of the tables have ? entries
        TypeError: Throws an error if the numerical columns end up with string value entries.

    Returns:
        A tuple with 3 values:
            - The tables dictionary.
            - The relation order between the tables.
            - The dataset metadata dictionary.
    """
    meta_path = data_dir / "dataset_meta.json" if meta_dir is None else meta_dir / "dataset_meta.json"
    with open(meta_path, "r") as f:
        dataset_meta = json.load(f)

    relation_order = dataset_meta["relation_order"]
    tables: Tables = {}

    for table, meta in dataset_meta["tables"].items():
        csv_path = data_dir / train_name
        train_df = pd.read_csv(csv_path)

        domain_path = data_dir / f"{table}_domain.json"
        with open(domain_path, "r") as domain_file:
            domain = json.load(domain_file)

        id_cols = [col for col in train_df.columns if "_id" in col]
        df_no_id = train_df.drop(columns=id_cols)
        info = get_info_from_domain(df_no_id, domain)

        tables[table] = Table(
            data=train_df,
            domain=domain,
            children=meta["children"],
            parents=meta["parents"],
            original_column_names=list(train_df.columns),
            original_data=train_df.copy(),
            info=info,
        )

        # Columns containing '?'
        question_mark_cols = (df_no_id == "?").any()
        if question_mark_cols.any():
            bad_cols = question_mark_cols[question_mark_cols].index.tolist()
            raise ValueError(f"Invalid values '?' detected in columns {bad_cols} of table '{table}'.")

        # Numeric columns containing strings
        num_cols = df_no_id.select_dtypes(include=["number"]).columns
        for col in num_cols:
            if df_no_id[col].map(lambda v: isinstance(v, str)).any():
                raise TypeError(f"Numeric column '{col}' in table '{table}' contains string values.")

    return tables, relation_order, dataset_meta


def save_results_and_plot_roc_curve(
    fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, all_results: list[dict[str, Any]], results_path: Path
) -> None:
    """
    Saves the ROC curve plot and results summary to the specified directory.

    Args:
        fpr: FPR values across a number of classification thresholds.
        tpr: TPR values across a number of classification thresholds.
        roc_auc: The roc_auc value for these curves.
        all_results: A collection of results
        results_path: Where to save the plots and the results values.
    """
    os.makedirs(results_path, exist_ok=True)
    plot_path = results_path / "roc_curve_models.png"

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Save results summary
    results_df = pd.DataFrame(all_results)
    results_summary_path = results_path / "results_summary.csv"
    results_df.to_csv(results_summary_path, index=False)
    log(INFO, f"✅ All runs completed. Results saved to {results_summary_path}")


def prepare_data_for_attack(
    model_indices: list[int], model_type: str, models_base_dir: Path, columns_for_deduplication: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepares data for an attack by merging and deduplicating datasets.

    Args:
        model_indices: List of model indices over which to iterate and for which to gather information.
        model_type: Name of the model type for which we're loading data.
        models_base_dir: Where the various models' data lives.
        columns_for_deduplication: Names of columns to use in de-duplicating the dataframes

    Raises:
        ValueError: Throws if the list of model indices is empty.
        ValueError: Throws if any of the dataframes to be de-duplicated do not have the specified columns in
            ``columns_for_deduplication``

    Returns:
        Tuple of three dataframes corresponding to the merged training data, challenge points, and challenge labels
        across the various models.
    """
    if len(model_indices) == 0:
        raise ValueError("The 'indices' list is empty. Please provide indices to process datasets.")

    df_merge_list = []
    df_challenge_list = []
    df_challenge_labels_list = []

    for model_index in model_indices:
        base_path = models_base_dir / f"{model_type}_{model_index}"
        df_merge_list.append(pd.read_csv(os.path.join(base_path, "train_with_id.csv")))
        df_challenge_list.append(pd.read_csv(os.path.join(base_path, "challenge_with_id.csv")))
        df_challenge_labels_list.append(pd.read_csv(os.path.join(base_path, "challenge_label.csv")))

    df_merge = pd.concat(df_merge_list, ignore_index=True)
    df_challenge = pd.concat(df_challenge_list, ignore_index=True)
    df_challenge_labels = pd.concat(df_challenge_labels_list, ignore_index=True)

    # Deduplicate the datasets once
    df_merge = df_merge.drop_duplicates(subset=columns_for_deduplication)
    df_challenge = df_challenge.drop_duplicates(subset=columns_for_deduplication)
    # TODO: Do we need to de-duplicate the labels dataframes as well?

    # Ensure all keys for deduplication exist in both DataFrames
    missing_keys_merge = [key for key in columns_for_deduplication if key not in df_merge.columns]
    missing_keys_challenge = [key for key in columns_for_deduplication if key not in df_challenge.columns]
    if missing_keys_merge or missing_keys_challenge:
        raise ValueError(f"Missing columns for deduplication: {missing_keys_merge + missing_keys_challenge}")

    df_merge_without_challenge = df_merge[
        ~df_merge.set_index(columns_for_deduplication).index.isin(
            df_challenge.set_index(columns_for_deduplication).index
        )
    ]

    return df_merge_without_challenge, df_challenge, df_challenge_labels


def evaluate_attack_performance(
    model_indices: list[int],
    description: str,
    tabddpm_data_dir: Path,
    model_type: str,
    predictions_file_name: str,
) -> dict[str, float]:
    """
    Load saved challenge prediction and label data for a collection of models, concatenate the predictions and labels
    together and then measure MIA attack success on the concatenation of these values.

    Args:
        model_indices: Model indices for each of the models prediction and label data that we'll be loading.
        description: A description of the models being loaded to be logged.
        tabddpm_data_dir: The top-level data directory where all of the model predictions and labels are stored.
        model_type: The model type/name being loaded. Together with the model index, this will form a folder name with
            the structure "{model_type}_{model_index}"
        predictions_file_name: Name of the prediction file output by all models. This will be the same for all models
            in question. (Must include file suffix)

    Raises:
        ValueError: If no model indices are provided.

    Returns:
        A dictionary containing the TPR@FPR=0.1 and the auc of the combined predictions. These are keyed by
        "max_tpr" and "roc_auc," respectively.
    """
    if len(model_indices) == 0:
        raise ValueError(f"Indices list is empty for {description}. Cannot evaluate attack performance.")

    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for model_number in tqdm(model_indices, desc=f"Evaluating on {description} models", unit="model"):
        model_folder = f"{model_type}_{model_number}"
        model_artifacts_path = tabddpm_data_dir / model_folder

        predictions.append(np.loadtxt(model_artifacts_path / predictions_file_name))
        labels.append(np.loadtxt(model_artifacts_path / "challenge_label.csv", skiprows=1))

    if len(predictions) == 0:
        raise ValueError("No predictions found!")

    predictions_arr = np.concatenate(predictions)
    solutions_arr = np.concatenate(labels)

    tpr_at_fpr = TprAtFpr.get_tpr_at_fpr(solutions_arr, predictions_arr)
    roc_auc = roc_auc_score(solutions_arr, predictions_arr)

    return {"max_tpr": tpr_at_fpr, "roc_auc": roc_auc}
