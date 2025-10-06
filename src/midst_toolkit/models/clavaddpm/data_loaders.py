import json
import os
from collections.abc import Generator
from logging import INFO
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import torch

from midst_toolkit.common.enumerations import DataSplit, DomainDataType, InfoDataType, TaskType
from midst_toolkit.common.logger import log
from midst_toolkit.models.clavaddpm.dataset import Dataset
from midst_toolkit.models.clavaddpm.enumerations import TargetType


def load_multi_table(
    data_dir: Path,
    verbose: bool = True,
    training_data_ratio: float = 1,
) -> tuple[dict[str, Any], list[tuple[str, str]], dict[str, Any]]:
    """
    Load the multi-table dataset from the data directory.

    Args:
        data_dir: The directory to load the dataset from.
        verbose: Whether to print verbose output. Optional, default is True.
        training_data_ratio: The ratio of the data to be used for training. Should be between 0 and 1.
            If it's == 1, it will only return the training set. Optional, default is 1.

    Returns:
        A tuple with 3 values:
            - The tables dictionary.
            - The relation order between the tables.
            - The dataset metadata dictionary.
    """
    with open(data_dir / "dataset_meta.json", "r") as f:
        dataset_meta = json.load(f)

    relation_order = dataset_meta["relation_order"]

    tables = {}

    for table, meta in dataset_meta["tables"].items():
        if (data_dir / "train.csv").exists():
            train_df = pd.read_csv(data_dir / "train.csv")
        else:
            train_df = pd.read_csv(data_dir / f"{table}.csv")

        with open(data_dir / f"{table}_domain.json", "r") as f:
            domain = json.load(f)

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
        info = get_info_from_domain(df_no_id, tables[table]["domain"])
        _, info = process_pipeline_data(
            table_name=table,
            data=df_no_id,
            info=info,
            training_data_ratio=training_data_ratio,
            save=False,
            verbose=verbose,
        )
        tables[table]["info"] = info

    return tables, relation_order, dataset_meta


def get_info_from_domain(data: pd.DataFrame, table_domain: dict[str, Any]) -> dict[str, Any]:
    """
    Get the information dictionary from the table domain dictionary.

    Args:
        data: The dataframe containing the data.
        table_domain: The table's domain dictionary containing metadata about the data columns.

    Returns:
        The information dictionary containing the following keys:
        - num_col_idx: The indices of the numerical columns.
        - cat_col_idx: The indices of the categorical columns.
        - target_col_idx: The indices of the target columns.
        - task_type: The type of the task.
        - column_names: The names of all the columns.
    """
    info: dict[str, Any] = {}
    info["num_col_idx"] = []
    info["cat_col_idx"] = []
    columns = data.columns.tolist()
    for i in range(len(columns)):
        if table_domain[columns[i]]["type"] == DomainDataType.DISCRETE.value:
            info["cat_col_idx"].append(i)
        else:
            info["num_col_idx"].append(i)

    info["target_col_idx"] = []
    info["task_type"] = None
    info["column_names"] = columns

    return info


def process_pipeline_data(
    # ruff: noqa: PLR0915, PLR0912
    table_name: str,
    data: pd.DataFrame,
    info: dict[str, Any],
    training_data_ratio: float = 0.9,
    save: bool = False,
    verbose: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """
    Processes the data to be sent through the pipeline.

    Will split the data into training and test sets (saving the data when specified),
    replace invalid and missing values, split the data sets categorical, numerical
    and target columns, and populate the information dictionary with additional
    metadata.

    Args:
        table_name: The name of the table. Used to name the files when saving the data.
        data: The dataframe containing the data.
        info: The information dictionary, retrieved from the get_info_from_domain function.
        training_data_ratio: The ratio of the data to be used for training. Should be between 0 and 1.
            If it's == 1, it will only return the training set. Optional, default is 0.9.
        save: Whether to save the data. Optional, default is False.
        verbose: Whether to print verbose output. Optional, default is True.

    Returns:
        A tuple with 2 values:
            - The data dictionary containing the following keys:
                - "df": The dataframe containing the data.
                    - DataSplit.TRAIN: The dataframe containing the training set.
                    - DataSplit.TEST: The dataframe containing the test set. It will be absent if ratio == 1.
                - "numpy": A dictionary with the numeric data, containing the keys:
                    - "x_num_train": The numeric data for the training set.
                    - "x_cat_train": The categorical data for the training set.
                    - "y_train": The target data for the training set.
                    - "x_num_test": The numeric data for the test set. It will be absent if ratio == 1.
                    - "x_cat_test": The categorical data for the test set. It will be absent if ratio == 1.
                    - "y_test": The target data for the test set. It will be absent if ratio == 1.
            - The information dictionary with updated values.
    """
    assert 0 < training_data_ratio <= 1, "Training data ratio must be between 0 and 1."
    if training_data_ratio == 1:
        log(INFO, "Training data ratio is 1, so the data will not be split into training and test sets.")

    num_samples = data.shape[0]

    column_names = info["column_names"] if info["column_names"] else data.columns.tolist()

    numerical_column_indices = info["num_col_idx"]
    categorical_column_indices = info["cat_col_idx"]
    target_columns_indices = info["target_col_idx"]

    index_mapping, inverse_index_mapping, index_to_name_mapping = get_column_name_mapping(
        data,
        numerical_column_indices,
        categorical_column_indices,
        column_names,
    )

    numerical_columns = [column_names[i] for i in numerical_column_indices]
    categorical_columns = [column_names[i] for i in categorical_column_indices]
    target_columns = [column_names[i] for i in target_columns_indices]

    # Train/ Test Split:
    # num_train_samples% for Training, (1 - num_test_samples)% for Testing
    # Validation set will be selected from Training set
    num_train_samples = int(num_samples * training_data_ratio)
    num_test_samples = num_samples - num_train_samples

    test_data: pd.DataFrame | None = None

    if training_data_ratio < 1:
        train_data, test_data, _ = train_test_split(data, categorical_columns, num_train_samples, num_test_samples)
    else:
        train_data = data.copy()

    train_data.columns = list(range(len(train_data.columns)))

    if training_data_ratio < 1:
        assert test_data is not None
        test_data.columns = list(range(len(test_data.columns)))

    columns_info: dict[Any, Any] = {}

    for column in numerical_column_indices:
        columns_info[column] = {}
        columns_info["type"] = InfoDataType.NUMERICAL.value
        columns_info["max"] = float(train_data[column].max())
        columns_info["min"] = float(train_data[column].min())

    for column in categorical_column_indices:
        columns_info[column] = {}
        columns_info["type"] = InfoDataType.CATEGORICAL.value
        columns_info["categorizes"] = list(set(train_data[column]))

    for column in target_columns_indices:
        if info["task_type"] == TaskType.REGRESSION.value:
            columns_info[column] = {}
            columns_info["type"] = InfoDataType.NUMERICAL.value
            columns_info["max"] = float(train_data[column].max())
            columns_info["min"] = float(train_data[column].min())
        else:
            columns_info[column] = {}
            columns_info["type"] = InfoDataType.CATEGORICAL.value
            columns_info["categorizes"] = list(set(train_data[column]))

    info["column_info"] = columns_info

    train_data.rename(columns=index_to_name_mapping, inplace=True)
    if training_data_ratio < 1:
        assert test_data is not None
        test_data.rename(columns=index_to_name_mapping, inplace=True)

    for col in numerical_columns:
        train_data.loc[train_data[col] == "?", col] = np.nan
    for col in categorical_columns:
        train_data.loc[train_data[col] == "?", col] = "nan"

    if training_data_ratio < 1:
        assert test_data is not None
        for col in numerical_columns:
            test_data.loc[test_data[col] == "?", col] = np.nan
        for col in categorical_columns:
            test_data.loc[test_data[col] == "?", col] = "nan"

    x_num_train = train_data[numerical_columns].to_numpy().astype(np.float32)
    x_cat_train = train_data[categorical_columns].to_numpy()
    y_train = train_data[target_columns].to_numpy()

    x_num_test: np.ndarray | None = None
    x_cat_test: np.ndarray | None = None
    y_test: np.ndarray | None = None

    if training_data_ratio < 1:
        assert test_data is not None
        x_num_test = test_data[numerical_columns].to_numpy().astype(np.float32)
        x_cat_test = test_data[categorical_columns].to_numpy()
        y_test = test_data[target_columns].to_numpy()

    if save:
        save_dir = f"data/{table_name}"
        np.save(f"{save_dir}/x_num_train.npy", x_num_train)
        np.save(f"{save_dir}/x_cat_train.npy", x_cat_train)
        np.save(f"{save_dir}/y_train.npy", y_train)

        if training_data_ratio < 1:
            assert x_num_test is not None and x_cat_test is not None and y_test is not None
            np.save(f"{save_dir}/x_num_test.npy", x_num_test)
            np.save(f"{save_dir}/x_cat_test.npy", x_cat_test)
            np.save(f"{save_dir}/y_test.npy", y_test)

    train_data[numerical_columns] = train_data[numerical_columns].astype(np.float32)

    if training_data_ratio < 1:
        assert test_data is not None
        test_data[numerical_columns] = test_data[numerical_columns].astype(np.float32)

    if save:
        train_data.to_csv(f"{save_dir}/train.csv", index=False)

        if training_data_ratio < 1:
            assert test_data is not None
            test_data.to_csv(f"{save_dir}/test.csv", index=False)

        if not os.path.exists(f"synthetic/{table_name}"):
            os.makedirs(f"synthetic/{table_name}")

        train_data.to_csv(f"synthetic/{table_name}/real.csv", index=False)

        if training_data_ratio < 1:
            assert test_data is not None
            test_data.to_csv(f"synthetic/{table_name}/test.csv", index=False)

    info["column_names"] = column_names
    info["train_num"] = train_data.shape[0]

    if training_data_ratio < 1:
        assert test_data is not None
        info["test_num"] = test_data.shape[0]

    info["idx_mapping"] = index_mapping
    info["inverse_idx_mapping"] = inverse_index_mapping
    info["idx_name_mapping"] = index_to_name_mapping

    metadata: dict[str, Any] = {"columns": {}}
    task_type = info["task_type"]
    numerical_column_indices = info["num_col_idx"]
    categorical_column_indices = info["cat_col_idx"]
    target_columns_indices = info["target_col_idx"]

    for i in numerical_column_indices:
        metadata["columns"][i] = {
            "sdtype": InfoDataType.NUMERICAL.value,
            "computer_representation": "Float",
        }

    for i in categorical_column_indices:
        metadata["columns"][i] = {"sdtype": InfoDataType.CATEGORICAL.value}

    if task_type == TaskType.REGRESSION.value:
        for i in target_columns_indices:
            metadata["columns"][i] = {
                "sdtype": InfoDataType.NUMERICAL.value,
                "computer_representation": "Float",
            }

    else:
        for i in target_columns_indices:
            metadata["columns"][i] = {"sdtype": InfoDataType.CATEGORICAL.value}

    info["metadata"] = metadata

    if save:
        with open(f"{save_dir}/info.json", "w") as file:
            json.dump(info, file, indent=4)

    if verbose:
        if training_data_ratio < 1:
            assert test_data is not None
            str_shape = f"Train dataframe shape: {train_data.shape}, Test dataframe shape: {test_data.shape}, Total dataframe shape: {data.shape}"
        else:
            str_shape = f"Table name: {table_name}, Total dataframe shape: {data.shape}"

        str_shape += f", Numerical data shape: {x_num_train.shape}"
        str_shape += f", Categorical data shape: {x_cat_train.shape}"
        log(INFO, str_shape)

    output_data: dict[str, dict[str, Any]] = {
        "df": {
            DataSplit.TRAIN.value: train_data,
        },
        "numpy": {
            "x_num_train": x_num_train,
            "x_cat_train": x_cat_train,
            "y_train": y_train,
        },
    }

    if training_data_ratio < 1:
        assert test_data is not None and x_num_test is not None and x_cat_test is not None and y_test is not None
        output_data["df"][DataSplit.TEST.value] = test_data
        output_data["numpy"]["x_num_test"] = x_num_test
        output_data["numpy"]["x_cat_test"] = x_cat_test
        output_data["numpy"]["y_test"] = y_test

    return output_data, info


def get_column_name_mapping(
    data: pd.DataFrame,
    numerical_columns_indices: list[int],
    categorical_column_indices: list[int],
    column_names: list[str] | None = None,
) -> tuple[dict[int, int], dict[int, int], dict[int, str]]:
    """
    Get the column name mappings.

    Will produce 3 mappings:
        - The mapping of the categorical and numerical columns from their original indices
            in the dataframe to their indices in the numerical_columns_indices and
            categorical_column_indices lists.
        - The inverse mapping of the above, i.e. the mapping from their indices in the
            numerical_columns_indices and categorical_column_indices lists to their original
            indices in the dataframe.
        - The mapping of the indices in the original dataframe to the column names for all columns.

    Args:
        data: The dataframe containing the data.
        numerical_columns_indices: The indices of the numerical columns.
        categorical_column_indices: The indices of the categorical columns.
        column_names: The names of the columns. Optional, default is None. If None,
            it will use the columns of the dataframe.

    Returns:
        A tuple with 3 values:
            - The mapping of the categorical and numerical columns from their original indices
            in the dataframe to their indices in the numerical_columns_indices and
            categorical_column_indices lists.
            - The inverse mapping of the above, i.e. the mapping from their indices in the
            numerical_columns_indices and categorical_column_indices lists to their original
            indices in the dataframe.
            - The mapping of the indices in the original dataframe to the column names for all columns.
    """
    if not column_names:
        column_names = data.columns.tolist()

    index_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(numerical_columns_indices)
    curr_target_idx = curr_cat_idx + len(categorical_column_indices)

    for idx in range(len(column_names)):
        if idx in numerical_columns_indices:
            index_mapping[idx] = curr_num_idx
            curr_num_idx += 1
        elif idx in categorical_column_indices:
            index_mapping[idx] = curr_cat_idx
            curr_cat_idx += 1
        else:
            index_mapping[idx] = curr_target_idx
            curr_target_idx += 1

    inverse_index_mapping = {}
    for k, v in index_mapping.items():
        inverse_index_mapping[v] = k

    index_to_name_mapping = {}

    for i in range(len(column_names)):
        index_to_name_mapping[i] = column_names[i]

    return index_mapping, inverse_index_mapping, index_to_name_mapping


# TODO: refactor this function so it doesn't run the risk of running indefinitely.
def train_test_split(
    data: pd.DataFrame,
    categorical_columns: list[str],
    num_train_samples: int = 0,
    num_test_samples: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Split the data into training and test sets.

    Will make the split in a way that both sets have all the values for the categorical
    columns represented.

    Args:
        data: The dataframe containing the data.
        categorical_columns: The names of the categorical columns.
        num_train_samples: The number of rows in the training set. Optional, default is 0.
        num_test_samples: The number of rows in the test set. Optional, default is 0.

    Returns:
        A tuple with 3 values:
            - The training dataframe.
            - The test dataframe.
            - The seed used by the random number generator to generate the split.
    """
    total_num = data.shape[0]
    idx = np.arange(total_num)

    seed = 1234

    while True:
        np.random.seed(seed)
        np.random.shuffle(idx)

        train_idx = idx[:num_train_samples]
        test_idx = idx[-num_test_samples:]

        train_df = data.loc[train_idx]
        test_df = data.loc[test_idx]

        flag = 0
        for i in categorical_columns:
            if len(set(train_df[i])) != len(set(data[i])):
                flag = 1
                break

        if flag == 0:
            break
        seed += 1

    return train_df, test_df, seed


class FastTensorDataLoader:
    def __init__(self, tensors: tuple[torch.Tensor, ...], batch_size: int = 32, shuffle: bool = False):
        """
        Initialize a FastTensorDataLoader.

        A DataLoader-like object for a set of tensors that can be much faster than
        TensorDataset + DataLoader because dataloader grabs individual indices of
        the dataset and calls cat (slow).
        Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6

        Args:
            tensors: a tuple of tensors to store. The first dimension for each tensor is the
                number of samples, and all tensors must have the same number of samples.
            batch_size: batch size to load. Optional, default is 32.
            shuffle: if True, shuffle the data *in-place* whenever an
                iterator is created out of this object. Optional, default is False.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors), (
            "All tensors must have the same amount of samples."
        )
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self) -> Self:
        """
        Define the iterator for the FastTensorDataLoader.

        Returns:
            The FastTensorDataLoader prepared for iteration.
        """
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]  # type: ignore[assignment]
        self.i = 0
        return self

    def __next__(self) -> tuple[torch.Tensor, ...]:
        """Get the next batch of data from the dataset.

        Returns:
            A tuple of tensors, one for each tensor in the FastTensorDataLoader.
        """
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self) -> int:
        """
        Get the number of batches in the dataset.

        Returns:
            (int) The number of batches in the dataset.
        """
        return self.n_batches


def prepare_fast_dataloader(
    dataset: Dataset,
    split: DataSplit,
    batch_size: int,
    target_type: TargetType = TargetType.FLOAT,
) -> Generator[tuple[torch.Tensor, ...]]:
    """
    Prepare a fast dataloader for the dataset.

    Args:
        dataset: The dataset to prepare the dataloader for.
        split: The split to prepare the dataloader for.
        batch_size: The batch size to use for the dataloader.
        target_type: The type of the target values. Default is TargetType.FLOAT.

    Returns:
        A generator of batches of data from the dataset.
    """
    if dataset.x_cat is not None:
        if dataset.x_num is not None:
            concatenated_features = np.concatenate([dataset.x_num[split.value], dataset.x_cat[split.value]], axis=1)
            x = torch.from_numpy(concatenated_features).float()
        else:
            x = torch.from_numpy(dataset.x_cat[split.value]).float()
    else:
        assert dataset.x_num is not None
        x = torch.from_numpy(dataset.x_num[split.value]).float()

    if target_type == TargetType.FLOAT:
        y = torch.from_numpy(dataset.y[split.value]).float()
    elif target_type == TargetType.LONG:
        y = torch.from_numpy(dataset.y[split.value]).long()
    else:
        raise ValueError(f"Unsupported target type: {target_type}")

    dataloader = FastTensorDataLoader((x, y), batch_size=batch_size, shuffle=(split == DataSplit.TRAIN))
    while True:
        yield from dataloader
