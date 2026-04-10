import json
from collections.abc import Generator
from dataclasses import dataclass
from logging import INFO
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from midst_toolkit.common.enumerations import ColumnType, ComputerRepresentation, DataSplit, DomainDataType, TaskType
from midst_toolkit.common.logger import log
from midst_toolkit.models.clavaddpm.dataset import Dataset
from midst_toolkit.models.clavaddpm.enumerations import RelationOrder, TargetType


@dataclass
class NumericalColumnInfo:
    max: float
    min: float


@dataclass
class CategoricalColumnInfo:
    categorizes: list[str]


@dataclass
class ColumnInfo:
    type: ColumnType
    info: NumericalColumnInfo | CategoricalColumnInfo


@dataclass
class ColumnMetadata:
    sdtype: ColumnType
    computer_representation: ComputerRepresentation | None = None


@dataclass
class DomainInfo:
    numerical_column_indices: list[int]
    categorical_column_indices: list[int]
    target_column_indices: list[int]
    task_type: TaskType | None
    column_names: list[str]
    columns_info: dict[str, ColumnInfo] | None = None
    train_num: int | None = None
    test_num: int | None = None
    metadata: dict[int, ColumnMetadata] | None = None


@dataclass
class Table:
    data: pd.DataFrame
    domain: dict[str, Any]
    children: list[str]
    parents: list[str]
    original_column_names: list[str]
    original_data: pd.DataFrame
    info: DomainInfo


Tables = dict[str, Table]


NO_PARENT_COLUMN_NAME = "placeholder"  # Value to assign to key column for tables with no parent


def load_tables(
    data_dir: Path,
    verbose: bool = True,
    training_data_ratio: float = 1,
    train_data: dict[str, pd.DataFrame] | None = None,
) -> tuple[Tables, RelationOrder, dict[str, Any]]:

    with open(data_dir / "dataset_meta.json", "r") as f:
        dataset_meta = json.load(f)

    relation_order = [tuple(relation) for relation in dataset_meta["relation_order"]]

    tables = {}

    for table, meta in dataset_meta["tables"].items():
        if train_data is not None:
            train_df = train_data[table]
        elif (data_dir / "train.csv").exists():
            train_df = pd.read_csv(data_dir / "train.csv")
        else:
            train_df = pd.read_csv(data_dir / f"{table}.csv")

        with open(data_dir / f"{table}_domain.json", "r") as f:
            domain = json.load(f)

        id_cols = [col for col in train_df.columns if "_id" in col]
        df_no_id = train_df.drop(columns=id_cols)
        info = process_pipeline_data(df_no_id, domain, training_data_ratio, verbose)

        tables[table] = Table(
            data=train_df,
            domain=domain,
            children=meta["children"],
            parents=meta["parents"],
            original_column_names=list(train_df.columns),
            original_data=train_df.copy(),
            info=info,
        )

    return tables, relation_order, dataset_meta


def get_info_from_domain(data: pd.DataFrame, table_domain: dict[str, Any]) -> DomainInfo:

    numerical_column_indices = []
    categorical_column_indices = []
    columns = data.columns.tolist()
    for i in range(len(columns)):
        if table_domain[columns[i]]["type"] == DomainDataType.DISCRETE.value:
            categorical_column_indices.append(i)
        else:
            numerical_column_indices.append(i)

    return DomainInfo(
        numerical_column_indices=numerical_column_indices,
        categorical_column_indices=categorical_column_indices,
        target_column_indices=[],
        task_type=None,
        column_names=columns,
    )


@dataclass
class DataFeatures:
    data: pd.DataFrame
    numerical_features: np.ndarray | None = None
    categorical_features: np.ndarray | None = None
    target_features: np.ndarray | None = None


@dataclass
class DataSplits:
    train_data: DataFeatures
    test_data: DataFeatures | None = None
    seed: int | None = None


def process_pipeline_data(
    data: pd.DataFrame,
    table_domain: dict[str, Any],
    training_data_ratio: float = 0.9,
    verbose: bool = True,
) -> DomainInfo:

    data_splits, info = _split_data_and_generate_info(data, table_domain, training_data_ratio)

    metadata: dict[int, ColumnMetadata] = {}

    for i in info.numerical_column_indices:
        metadata[i] = ColumnMetadata(
            sdtype=ColumnType.NUMERICAL,
            computer_representation=ComputerRepresentation.FLOAT,
        )

    for i in info.categorical_column_indices:
        metadata[i] = ColumnMetadata(sdtype=ColumnType.CATEGORICAL)

    if info.task_type == TaskType.REGRESSION:
        for i in info.target_column_indices:
            metadata[i] = ColumnMetadata(
                sdtype=ColumnType.NUMERICAL,
                computer_representation=ComputerRepresentation.FLOAT,
            )

    else:
        for i in info.target_column_indices:
            metadata[i] = ColumnMetadata(sdtype=ColumnType.CATEGORICAL)

    info.metadata = metadata

    if verbose:
        log(INFO, f"Train dataframe shape: {data_splits.train_data.data.shape}")
        if data_splits.test_data is not None:
            log(INFO, f"Test dataframe shape: {data_splits.test_data.data.shape}")
        log(INFO, f"Total dataframe shape: {data.shape}")

        assert data_splits.train_data.numerical_features is not None
        assert data_splits.train_data.categorical_features is not None
        log(INFO, f"Numerical data shape: {data_splits.train_data.numerical_features.shape}")
        log(INFO, f"Categorical data shape: {data_splits.train_data.categorical_features.shape}")

    return info


def _get_columns_info(
    train_data: pd.DataFrame,
    numerical_column_indices: list[int],
    categorical_column_indices: list[int],
    target_column_indices: list[int],
    task_type: TaskType | None,
) -> dict[str, ColumnInfo]:

    columns_info: dict[str, ColumnInfo] = {}

    for column in numerical_column_indices:
        column_name = train_data.columns[column]
        columns_info[column_name] = ColumnInfo(
            type=ColumnType.NUMERICAL,
            info=NumericalColumnInfo(
                max=float(train_data[column_name].max()),
                min=float(train_data[column_name].min()),
            ),
        )

    for column in categorical_column_indices:
        column_name = train_data.columns[column]
        columns_info[column_name] = ColumnInfo(
            type=ColumnType.CATEGORICAL,
            info=CategoricalColumnInfo(
                categorizes=list(set(train_data[column_name])),
            ),
        )

    for column in target_column_indices:
        if task_type == TaskType.REGRESSION:
            column_name = train_data.columns[column]
            columns_info[column_name] = ColumnInfo(
                type=ColumnType.NUMERICAL,
                info=NumericalColumnInfo(
                    max=float(train_data[column_name].max()),
                    min=float(train_data[column_name].min()),
                ),
            )
        else:
            column_name = train_data.columns[column]
            columns_info[column_name] = ColumnInfo(
                type=ColumnType.CATEGORICAL,
                info=CategoricalColumnInfo(
                    categorizes=list(set(train_data[column_name])),
                ),
            )

    return columns_info


def _split_data_and_generate_info(
    data: pd.DataFrame,
    table_domain: dict[str, Any],
    training_data_ratio: float,
) -> tuple[DataSplits, DomainInfo]:

    info = get_info_from_domain(data, table_domain)

    numerical_column_names = [info.column_names[i] for i in info.numerical_column_indices]
    categorical_column_names = [info.column_names[i] for i in info.categorical_column_indices]
    target_column_names = [info.column_names[i] for i in info.target_column_indices]

    # Splitting the data into training and test sets
    data_splits = train_test_split(data, categorical_column_names, training_data_ratio)

    # Populating the column info into the info dictionary
    info.columns_info = _get_columns_info(
        data_splits.train_data.data,
        info.numerical_column_indices,
        info.categorical_column_indices,
        info.target_column_indices,
        info.task_type,
    )

    # Replace the invalid and missing values with np.nan for the numerical columns
    # and "nan" for the categorical columns
    for column_name in numerical_column_names:
        column_data = data_splits.train_data.data[column_name]
        data_splits.train_data.data.loc[column_data == "?", column_name] = np.nan
    for column_name in categorical_column_names:
        column_data = data_splits.train_data.data[column_name]
        data_splits.train_data.data.loc[column_data == "?", column_name] = "nan"

    if data_splits.test_data is not None:
        for column_name in numerical_column_names:
            column_data = data_splits.test_data.data[column_name]
            data_splits.test_data.data.loc[column_data == "?", column_name] = np.nan
        for column_name in categorical_column_names:
            column_data = data_splits.test_data.data[column_name]
            data_splits.test_data.data.loc[column_data == "?", column_name] = "nan"

    # Extract the numerical, categorical and target features
    # and convert them to numpy arrays
    numerical_features = data_splits.train_data.data[numerical_column_names].to_numpy().astype(np.float32)
    data_splits.train_data.numerical_features = numerical_features
    categorical_features = data_splits.train_data.data[categorical_column_names].to_numpy()
    data_splits.train_data.categorical_features = categorical_features
    target_features = data_splits.train_data.data[target_column_names].to_numpy()
    data_splits.train_data.target_features = target_features

    if data_splits.test_data is not None:
        numerical_features = data_splits.test_data.data[numerical_column_names].to_numpy().astype(np.float32)
        data_splits.test_data.numerical_features = numerical_features
        categorical_features = data_splits.test_data.data[categorical_column_names].to_numpy()
        data_splits.test_data.categorical_features = categorical_features
        target_features = data_splits.test_data.data[target_column_names].to_numpy()
        data_splits.test_data.target_features = target_features

    # Making sure the numerical data is float
    numerical_data_as_float = data_splits.train_data.data[numerical_column_names].astype(np.float32)
    data_splits.train_data.data[numerical_column_names] = numerical_data_as_float

    if data_splits.test_data is not None:
        numerical_data_as_float = data_splits.test_data.data[numerical_column_names].astype(np.float32)
        data_splits.test_data.data[numerical_column_names] = numerical_data_as_float

    # Populating the rest of the info dictionary
    info.train_num = data_splits.train_data.data.shape[0]

    if data_splits.test_data is not None:
        info.test_num = data_splits.test_data.data.shape[0]

    return data_splits, info


# TODO: refactor this function so it doesn't run the risk of running indefinitely.
def train_test_split(
    data: pd.DataFrame,
    categorical_columns: list[str],
    training_data_ratio: float = 0.9,
) -> DataSplits:

    assert 0 < training_data_ratio <= 1, "Training data ratio must be between 0 and 1."
    if training_data_ratio == 1:
        log(INFO, "Training data ratio is 1, so the data will not be split into training and test sets.")

    if training_data_ratio == 1:
        return DataSplits(train_data=DataFeatures(data=data.copy()))

    # Train/ Test Split:# Train/ Test Split:
    # num_train_samples% for Training, (1 - num_test_samples)% for Testing
    # Validation set will be selected from Training set
    num_samples = data.shape[0]
    num_train_samples = int(num_samples * training_data_ratio)
    num_test_samples = num_samples - num_train_samples

    indices = np.arange(num_samples)
    current_seed = 1234
    while True:
        np.random.seed(current_seed)
        np.random.shuffle(indices)

        train_indices = indices[:num_train_samples]
        test_indices = indices[-num_test_samples:]

        train_data = data.loc[train_indices]
        test_data = data.loc[test_indices]

        stop = True
        for i in categorical_columns:
            if len(set(train_data[i])) != len(set(data[i])):
                stop = False
                break

        if stop:
            break

        current_seed += 1

    return DataSplits(
        train_data=DataFeatures(data=train_data),
        test_data=DataFeatures(data=test_data),
        seed=current_seed,
    )


class FastTensorDataLoader:
    def __init__(self, tensors: list[Tensor], batch_size: int = 32, shuffle: bool = False):

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
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self) -> list[Tensor]:
        """Get the next batch of data from the dataset.

        Returns:
            A list of tensors, one for each tensor in the FastTensorDataLoader.
        """
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = [t[self.i : self.i + self.batch_size] for t in self.tensors]
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
) -> Generator[list[Tensor]]:
   
    if dataset.categorical_features is not None:
        if dataset.numerical_features is not None:
            concatenated_features = np.concatenate(
                [dataset.numerical_features[split.value], dataset.categorical_features[split.value]],
                axis=1,
            )
            features = torch.from_numpy(concatenated_features).float()
        else:
            features = torch.from_numpy(dataset.categorical_features[split.value]).float()
    else:
        assert dataset.numerical_features is not None
        features = torch.from_numpy(dataset.numerical_features[split.value]).float()

    if target_type == TargetType.FLOAT:
        target = torch.from_numpy(dataset.target[split.value]).float()
    elif target_type == TargetType.LONG:
        target = torch.from_numpy(dataset.target[split.value]).long()
    else:
        raise ValueError(f"Unsupported target type: {target_type}")

    dataloader = FastTensorDataLoader([features, target], batch_size=batch_size, shuffle=(split == DataSplit.TRAIN))

    while True:
        yield from dataloader
