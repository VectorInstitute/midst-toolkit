import json
import os
from collections.abc import Generator
from logging import INFO
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from midst_toolkit.common.logger import log
from midst_toolkit.models.clavaddpm.dataset import Dataset


def load_multi_table(
    data_dir: str, verbose: bool = True
) -> tuple[dict[str, Any], list[tuple[str, str]], dict[str, Any]]:
    """
    Load the multi-table dataset from the data directory.

    Args:
        data_dir: The directory to load the dataset from.
        verbose: Whether to print verbose output. Optional, default is True.

    Returns:
        A tuple with 3 values:
            - The tables dictionary.
            - The relation order between the tables.
            - The dataset metadata dictionary.
    """
    dataset_meta = json.load(open(os.path.join(data_dir, "dataset_meta.json"), "r"))

    relation_order = dataset_meta["relation_order"]

    tables = {}

    for table, meta in dataset_meta["tables"].items():
        if os.path.exists(os.path.join(data_dir, "train.csv")):
            train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        else:
            train_df = pd.read_csv(os.path.join(data_dir, f"{table}.csv"))
        tables[table] = {
            "df": train_df,
            "domain": json.load(open(os.path.join(data_dir, f"{table}_domain.json"))),
            # ruff: noqa: SIM115
            "children": meta["children"],
            "parents": meta["parents"],
        }
        tables[table]["original_cols"] = list(tables[table]["df"].columns)
        tables[table]["original_df"] = tables[table]["df"].copy()
        id_cols = [col for col in tables[table]["df"].columns if "_id" in col]
        df_no_id = tables[table]["df"].drop(columns=id_cols)
        info = get_info_from_domain(df_no_id, tables[table]["domain"])
        _, info = pipeline_process_data(
            name=table,
            data_df=df_no_id,
            info=info,
            ratio=1,
            save=False,
            verbose=verbose,
        )
        tables[table]["info"] = info

    return tables, relation_order, dataset_meta


def get_info_from_domain(data_df: pd.DataFrame, domain_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Get the information dictionaryfrom the domain dictionary.

    Args:
        data_df: The dataframe containing the data.
        domain_dict: The domain dictionary containing metadata about the data columns.

    Returns:
        The information dictionary containing the following keys:
        - num_col_idx: The indices of the numerical columns.
        - cat_col_idx: The indices of the categorical columns.
        - target_col_idx: The indices of the target columns.
        - task_type: The type of the task.
        - column_names: The names of the columns.
    """
    info: dict[str, Any] = {}
    info["num_col_idx"] = []
    info["cat_col_idx"] = []
    columns = data_df.columns.tolist()
    for i in range(len(columns)):
        if domain_dict[columns[i]]["type"] == "discrete":
            info["cat_col_idx"].append(i)
        else:
            info["num_col_idx"].append(i)

    info["target_col_idx"] = []
    info["task_type"] = "None"
    info["column_names"] = columns

    return info


def pipeline_process_data(
    # ruff: noqa: PLR0915, PLR0912
    name: str,
    data_df: pd.DataFrame,
    info: dict[str, Any],
    ratio: float = 0.9,
    save: bool = False,
    verbose: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Process the data through the pipeline.

    Args:
        name: The name of the table.
        data_df: The dataframe containing the data.
        info: The information dictionary, retrieved from the get_info_from_domain function.
        ratio: The ratio of the data to be used for training. Optional, default is 0.9.
        save: Whether to save the data. Optional, default is False.
        verbose: Whether to print verbose output. Optional, default is True.

    Returns:
        A tuple with 2 values:
            - The data dictionary containing the following keys:
                - df: The dataframe containing the data.
                - numpy: A dictionary with the numeric data, containing the keys:
                    - X_num_train: The numeric data for the training set.
                    - X_cat_train: The categorical data for the training set.
                    - y_train: The target data for the training set.
                    - X_num_test: The numeric data for the test set.
                    - X_cat_test: The categorical data for the test set.
                    - y_test: The target data for the test set.
            - The information dictionary with updated values.
    """
    num_data = data_df.shape[0]

    column_names = info["column_names"] if info["column_names"] else data_df.columns.tolist()

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(
        data_df, num_col_idx, cat_col_idx, target_col_idx, column_names
    )

    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]
    target_columns = [column_names[i] for i in target_col_idx]

    # Train/ Test Split, 90% Training, 10% Testing (Validation set will be selected from Training set)
    num_train = int(num_data * ratio)
    num_test = num_data - num_train

    if ratio < 1:
        train_df, test_df, seed = train_val_test_split(data_df, cat_columns, num_train, num_test)
    else:
        train_df = data_df.copy()

    train_df.columns = range(len(train_df.columns))

    if ratio < 1:
        test_df.columns = range(len(test_df.columns))

    col_info: dict[Any, Any] = {}

    for col_idx in num_col_idx:
        col_info[col_idx] = {}
        col_info["type"] = "numerical"
        col_info["max"] = float(train_df[col_idx].max())
        col_info["min"] = float(train_df[col_idx].min())

    for col_idx in cat_col_idx:
        col_info[col_idx] = {}
        col_info["type"] = "categorical"
        col_info["categorizes"] = list(set(train_df[col_idx]))

    for col_idx in target_col_idx:
        if info["task_type"] == "regression":
            col_info[col_idx] = {}
            col_info["type"] = "numerical"
            col_info["max"] = float(train_df[col_idx].max())
            col_info["min"] = float(train_df[col_idx].min())
        else:
            col_info[col_idx] = {}
            col_info["type"] = "categorical"
            col_info["categorizes"] = list(set(train_df[col_idx]))

    info["column_info"] = col_info

    train_df.rename(columns=idx_name_mapping, inplace=True)
    if ratio < 1:
        test_df.rename(columns=idx_name_mapping, inplace=True)

    for col in num_columns:
        train_df.loc[train_df[col] == "?", col] = np.nan
    for col in cat_columns:
        train_df.loc[train_df[col] == "?", col] = "nan"

    if ratio < 1:
        for col in num_columns:
            test_df.loc[test_df[col] == "?", col] = np.nan
        for col in cat_columns:
            test_df.loc[test_df[col] == "?", col] = "nan"

    X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
    X_cat_train = train_df[cat_columns].to_numpy()
    y_train = train_df[target_columns].to_numpy()

    if ratio < 1:
        X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
        X_cat_test = test_df[cat_columns].to_numpy()
        y_test = test_df[target_columns].to_numpy()

    if save:
        save_dir = f"data/{name}"
        np.save(f"{save_dir}/X_num_train.npy", X_num_train)
        np.save(f"{save_dir}/X_cat_train.npy", X_cat_train)
        np.save(f"{save_dir}/y_train.npy", y_train)

        if ratio < 1:
            np.save(f"{save_dir}/X_num_test.npy", X_num_test)
            np.save(f"{save_dir}/X_cat_test.npy", X_cat_test)
            np.save(f"{save_dir}/y_test.npy", y_test)

    train_df[num_columns] = train_df[num_columns].astype(np.float32)

    if ratio < 1:
        test_df[num_columns] = test_df[num_columns].astype(np.float32)

    if save:
        train_df.to_csv(f"{save_dir}/train.csv", index=False)

        if ratio < 1:
            test_df.to_csv(f"{save_dir}/test.csv", index=False)

        if not os.path.exists(f"synthetic/{name}"):
            os.makedirs(f"synthetic/{name}")

        train_df.to_csv(f"synthetic/{name}/real.csv", index=False)

        if ratio < 1:
            test_df.to_csv(f"synthetic/{name}/test.csv", index=False)

    info["column_names"] = column_names
    info["train_num"] = train_df.shape[0]

    if ratio < 1:
        info["test_num"] = test_df.shape[0]

    info["idx_mapping"] = idx_mapping
    info["inverse_idx_mapping"] = inverse_idx_mapping
    info["idx_name_mapping"] = idx_name_mapping

    metadata: dict[str, Any] = {"columns": {}}
    task_type = info["task_type"]
    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    for i in num_col_idx:
        metadata["columns"][i] = {}
        metadata["columns"][i]["sdtype"] = "numerical"
        metadata["columns"][i]["computer_representation"] = "Float"

    for i in cat_col_idx:
        metadata["columns"][i] = {}
        metadata["columns"][i]["sdtype"] = "categorical"

    if task_type == "regression":
        for i in target_col_idx:
            metadata["columns"][i] = {}
            metadata["columns"][i]["sdtype"] = "numerical"
            metadata["columns"][i]["computer_representation"] = "Float"

    else:
        for i in target_col_idx:
            metadata["columns"][i] = {}
            metadata["columns"][i]["sdtype"] = "categorical"

    info["metadata"] = metadata

    if save:
        with open(f"{save_dir}/info.json", "w") as file:
            json.dump(info, file, indent=4)

    if verbose:
        if ratio < 1:
            str_shape = "Train dataframe shape: {}, Test dataframe shape: {}, Total dataframe shape: {}".format(
                train_df.shape, test_df.shape, data_df.shape
            )
        else:
            str_shape = "Table name: {}, Total dataframe shape: {}".format(name, data_df.shape)

        str_shape += ", Numerical data shape: {}".format(X_num_train.shape)
        str_shape += ", Categorical data shape: {}".format(X_cat_train.shape)
        log(INFO, str_shape)

    data = {
        "df": {"train": train_df},
        "numpy": {
            "X_num_train": X_num_train,
            "X_cat_train": X_cat_train,
            "y_train": y_train,
        },
    }

    if ratio < 1:
        data["df"]["test"] = test_df
        data["numpy"]["X_num_test"] = X_num_test
        data["numpy"]["X_cat_test"] = X_cat_test
        data["numpy"]["y_test"] = y_test

    return data, info


def get_column_name_mapping(
    data_df: pd.DataFrame,
    num_col_idx: list[int],
    cat_col_idx: list[int],
    target_col_idx: list[int],
    column_names: list[str] | None = None,
) -> tuple[dict[int, int], dict[int, int], dict[int, str]]:
    """
    Get the column name mapping.

    Args:
        data_df: The dataframe containing the data.
        num_col_idx: The indices of the numerical columns.
        cat_col_idx: The indices of the categorical columns.
        target_col_idx: The indices of the target columns.
        column_names: The names of the columns.

    Returns:
        A tuple with 3 values:
            - The mapping of the categorical and numerical columns to the indices.
            - The mapping of the column names to the indices.
            - The mapping of all the indices to the column names.
    """
    if not column_names:
        column_names = data_df.columns.tolist()

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):
        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1

    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k

    idx_name_mapping = {}

    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def train_val_test_split(
    data_df: pd.DataFrame,
    cat_columns: list[str],
    num_train: int = 0,
    num_test: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Split the data into training and test sets.

    Args:
        data_df: The dataframe containing the data.
        cat_columns: The names of the categorical columns.
        num_train: The number of rows in the training set. Optional, default is 0.
        num_test: The number of rows in the test set. Optional, default is 0.

    Returns:
        A tuple with 3 values:
            - The training dataframe.
            - The test dataframe.
            - The seed used for the random number generator.
    """
    total_num = data_df.shape[0]
    idx = np.arange(total_num)

    seed = 1234

    while True:
        np.random.seed(seed)
        np.random.shuffle(idx)

        train_idx = idx[:num_train]
        test_idx = idx[-num_test:]

        train_df = data_df.loc[train_idx]
        test_df = data_df.loc[test_idx]

        flag = 0
        for i in cat_columns:
            if len(set(train_df[i])) != len(set(data_df[i])):
                flag = 1
                break

        if flag == 0:
            break
        seed += 1

    return train_df, test_df, seed


class FastTensorDataLoader:
    """
    Defines a faster dataloader for PyTorch tensors.

    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors: torch.Tensor, batch_size: int = 32, shuffle: bool = False):
        """
        Initialize a FastTensorDataLoader.

        Args:
            *tensors: tensors to store. Must have the same length @ dim 0.
            batch_size: batch size to load.
            shuffle: if True, shuffle the data *in-place* whenever an
                iterator is created out of this object.
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

    def __iter__(self):
        """Defines the iterator for the FastTensorDataLoader."""
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]  # type: ignore[assignment]
        self.i = 0
        return self

    def __next__(self):
        """Get the next batch of data from the dataset."""
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        """Get the number of batches in the dataset."""
        return self.n_batches


def prepare_fast_dataloader(
    dataset: Dataset,
    split: Literal["train", "val", "test"],
    batch_size: int,
    y_type: str = "float",
) -> Generator[tuple[torch.Tensor, ...]]:
    """
    Prepare a fast dataloader for the dataset.

    Args:
        dataset: The dataset to prepare the dataloader for.
        split: The split to prepare the dataloader for.
        batch_size: The batch size to use for the dataloader.
        y_type: The type of the target values. Can be "float" or "long". Default is "float".

    Returns:
        A generator of batches of data from the dataset.
    """
    if dataset.X_cat is not None:
        if dataset.X_num is not None:
            X = torch.from_numpy(np.concatenate([dataset.X_num[split], dataset.X_cat[split]], axis=1)).float()
        else:
            X = torch.from_numpy(dataset.X_cat[split]).float()
    else:
        assert dataset.X_num is not None
        X = torch.from_numpy(dataset.X_num[split]).float()
    y = torch.from_numpy(dataset.y[split]).float() if y_type == "float" else torch.from_numpy(dataset.y[split]).long()
    dataloader = FastTensorDataLoader(X, y, batch_size=batch_size, shuffle=(split == "train"))
    while True:
        yield from dataloader
