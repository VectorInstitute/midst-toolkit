# ruff: noqa: D102, D105, D103, D200
# mypy: disable-error-code=no-untyped-def
# mypy: disable-error-code=has-type
# mypy: disable-error-code=index
# mypy: disable-error-code=attr-defined
# mypy: disable-error-code=assignment
from __future__ import annotations

import enum
import io
import json
import os
import pickle
from pathlib import Path

# at very top of file (optional but helpful)
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

from midst_toolkit.models.clavaddpm.dataset import Dataset


Arraydict = dict[str, np.ndarray]
Tensordict = dict[str, torch.Tensor]


CAT_MISSING_VALUE = "__nan__"
CAT_RARE_VALUE = "__rare__"
Normalization = Literal["standard", "quantile", "minmax"]
NumNanPolicy = Literal["drop-rows", "mean"]
CatNanPolicy = Literal["most_frequent"]
CatEncoding = Literal["one-hot", "counter"]
YPolicy = Literal["default"]


class TaskType(enum.Enum):
    BINCLASS = "binclass"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"

    def __str__(self) -> str:
        return self.value


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        # Force CUDA tensors to load on CPU
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def load_multi_table_customized(data_dir, meta_dir=None, train_name="train.csv", verbose=True):
    if meta_dir is None:
        meta_path = os.path.join(data_dir, "dataset_meta.json")
    else:
        meta_path = os.path.join(meta_dir, "dataset_meta.json")
    with open(meta_path, "r") as f:
        dataset_meta = json.load(f)

    relation_order = dataset_meta["relation_order"]
    tables = {}

    for table, meta in dataset_meta["tables"].items():
        csv_path = os.path.join(data_dir, train_name)
        if os.path.exists(csv_path):
            train_df = pd.read_csv(csv_path)
        else:
            raise ValueError(f"CSV file missing: {csv_path}")

        domain_path = os.path.join(data_dir, f"{table}_domain.json")
        with open(domain_path, "r") as domain_file:
            domain = json.load(domain_file)
        tables[table] = {
            "df": train_df,
            "domain": domain,
            "children": meta["children"],
            "parents": meta["parents"],
        }
        tables[table]["original_cols"] = list(tables[table]["df"].columns)
        tables[table]["original_df"] = tables[table]["df"].copy()
        id_cols = [col for col in tables[table]["df"].columns if "_id" in col]
        df_no_id = tables[table]["df"].drop(columns=id_cols)

        # Columns containing '?'
        qmark_cols = (df_no_id == "?").any()
        if qmark_cols.any():
            bad_cols = qmark_cols[qmark_cols].index.tolist()
            raise ValueError(f"Invalid values '?' detected in columns {bad_cols} of table '{table}'.")

        # Numeric columns containing strings
        num_cols = df_no_id.select_dtypes(include=["number"]).columns
        for col in num_cols:
            if df_no_id[col].map(lambda v: isinstance(v, str)).any():
                raise TypeError(f"Numeric column '{col}' in table '{table}' contains string values.")

    return tables, relation_order, dataset_meta


def save_results_and_plot_roc_curve(
    fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, all_results: list[dict[str, Any]], results_path: str
) -> None:
    """
    Saves the ROC curve plot and results summary to the specified directory.
    """
    os.makedirs(results_path, exist_ok=True)
    plot_filename = "roc_curve_models.png"
    plot_path = os.path.join(results_path, plot_filename)

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
    results_summary_path = os.path.join(results_path, "results_summary.csv")
    results_df.to_csv(results_summary_path, index=False)
    print(f"✅ All runs completed. Results saved to {results_summary_path}")


def prepare_data_for_attack(indices, model_type, models_base_dir, keys_for_deduplication):
    """
    Prepares data for an attack by merging and deduplicating datasets.
    """
    if not indices:
        raise ValueError("The 'indices' list is empty. Please provide indices to process datasets.")

    df_merge_list = []
    df_challenge_list = []

    for t in indices:
        base_path = models_base_dir / f"{model_type}_{t}"
        df_merge_list.append(pd.read_csv(os.path.join(base_path, "train_with_id.csv")))
        df_challenge_list.append(pd.read_csv(os.path.join(base_path, "challenge_with_id.csv")))
        df_challenge_labels = pd.read_csv(os.path.join(base_path, "challenge_label.csv"))

    df_merge = pd.concat(df_merge_list, ignore_index=True)
    df_challenge = pd.concat(df_challenge_list, ignore_index=True)

    # Deduplicate the datasets once
    df_merge = df_merge.drop_duplicates(subset=keys_for_deduplication)
    df_challenge = df_challenge.drop_duplicates(subset=keys_for_deduplication)

    # Ensure all keys for deduplication exist in both DataFrames
    missing_keys_merge = [key for key in keys_for_deduplication if key not in df_merge.columns]
    missing_keys_challenge = [key for key in keys_for_deduplication if key not in df_challenge.columns]
    if missing_keys_merge or missing_keys_challenge:
        raise ValueError(f"Missing columns for deduplication: {missing_keys_merge + missing_keys_challenge}")

    df_merge_without_challenge = df_merge[
        ~df_merge.set_index(keys_for_deduplication).index.isin(df_challenge.set_index(keys_for_deduplication).index)
    ]

    return df_merge_without_challenge, df_challenge, df_challenge_labels


def get_tpr_at_fpr(true_membership: list[int], predictions: list[float], max_fpr: float = 0.1) -> float:
    """
    Calculates the best True Positive Rate when the False Positive Rate is at most `max_fpr`.
    """
    fpr, tpr, _ = roc_curve(true_membership, predictions)
    valid_tpr = tpr[fpr <= max_fpr]
    if len(valid_tpr) == 0:
        return 0.0
    return float(max(valid_tpr))


def evaluate_attack_performance(
    indices: list[int] | None,
    description: str,
    tabddpm_data_dir: Path,
    model_type: str,
    predictions_file_name: str,
) -> dict[str, float]:
    if indices is None or len(indices) == 0:
        raise ValueError(f"Indices list is empty for {description}. Cannot evaluate attack performance.")

    predictions: list[np.ndarray] = []
    solutions: list[np.ndarray] = []
    for model_number in tqdm(indices, desc=f"Evaluating on {description} models", unit="model"):
        model_folder = f"{model_type}_{model_number}"
        path = tabddpm_data_dir / model_folder
        try:
            predictions.append(np.loadtxt(path / predictions_file_name))
            solutions.append(np.loadtxt(path / "challenge_label.csv", skiprows=1))
        except FileNotFoundError:
            print(f"Warning: Prediction or label file not found for model {model_number}. Skipping.")
            continue

    if not predictions:
        print(f"No predictions found for {description} models.")
        return {"max_tpr": 0.0, "roc_auc": 0.0}

    predictions_arr = np.concatenate(predictions)
    solutions_arr = np.concatenate(solutions)

    tpr_at_fpr = get_tpr_at_fpr(solutions_arr, predictions_arr)
    fpr, tpr, _ = roc_curve(solutions_arr, predictions_arr)
    roc_auc = auc(fpr, tpr)

    return {"max_tpr": tpr_at_fpr, "roc_auc": roc_auc}


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than TensorDataset + DataLoader because
    dataloader grabs individual indices of the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6.
    """

    def __init__(self, *tensors: torch.Tensor, batch_size: int = 32, shuffle: bool = False) -> None:
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self) -> "FastTensorDataLoader":
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self) -> tuple[torch.Tensor, ...]:
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self) -> int:
        return self.n_batches


def prepare_fast_dataloader(dataset: Dataset, split: str, batch_size: int, y_type: str = "float"):
    """
    Prepares and yields batches of data from a given dataset for a specified split using FastTensorDataLoader.

    This function combines numerical and categorical features (if available) for the specified split,
    converts them to PyTorch tensors, and creates a dataloader that yields batches indefinitely.
    The target variable can be returned as either float or long tensors based on `y_type`.

    Args:
        dataset (Dataset): The dataset object containing numerical and/or categorical features and targets.
        split (str): The split of the dataset to use (e.g., "train", "val", "test").
        batch_size (int): The number of samples per batch.
        y_type (str, optional): The type of the target tensor, either "float" or "long". Defaults to "float".

    Yields:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing a batch of features and corresponding targets.

    Note:
        - If both numerical and categorical features are present, they are concatenated along the feature axis.
        - The dataloader shuffles data only if the split is "train".
        - The generator yields batches indefinitely.
    """
    if dataset.categorical_features is not None:
        if dataset.numerical_features is not None:
            x = torch.from_numpy(
                np.concatenate([dataset.numerical_features[split], dataset.categorical_features[split]], axis=1)
            ).float()
        else:
            x = torch.from_numpy(dataset.categorical_features[split]).float()
    else:
        x = torch.from_numpy(dataset.numerical_features[split]).float()
    if y_type == "float":
        y = torch.from_numpy(dataset.target[split]).float()
    else:
        y = torch.from_numpy(dataset.target[split]).long()
    dataloader = FastTensorDataLoader(x, y, batch_size=batch_size, shuffle=(split == "train"))
    while True:
        yield from dataloader
