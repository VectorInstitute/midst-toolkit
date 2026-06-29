import json
from logging import INFO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from midst_toolkit.common.enumerations import TaskType
from midst_toolkit.common.logger import log


def get_processed_data_dir(data_dir: Path) -> Path:
    """Get the processed data directory.

    Args:
        data_dir: The data directory.

    Returns:
        The processed data directory.
    """
    return data_dir / "processed_data"


def get_column_name_mapping(
    data_df: pd.DataFrame,
    num_col_idx: list[int],
    cat_col_idx: list[int],
    target_col_idx: list[int],
    column_names: list[str] | None = None,
) -> tuple[dict[int, int], dict[int, int], dict[int, str]]:
    """Get the column name mappings for preprocessing.

    Args:
        data_df: The data frame.
        num_col_idx: The index of the numerical columns.
        cat_col_idx: The index of the categorical columns.
        target_col_idx: The index of the target column.
        column_names: The names of the columns. If not provided, the column names
            will be extracted from the data frame.

    Returns:
        A tuple of the index mapping, the inverse index mapping and the name mapping.
    """
    if not column_names:
        column_names = data_df.columns.tolist()

    column_names_np = np.array(column_names)

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names_np)):
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

    for i in range(len(column_names_np)):
        idx_name_mapping[int(i)] = column_names_np[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


# TODO: refactor this function, possibly re-use the one we refactored before that looked very similar:
# midst_toolkit/models/clavaddpm/data_loaders.py::train_test_split
def train_val_test_split(
    data_df: pd.DataFrame,
    cat_columns: list[str],
    num_train: int = 0,
    num_test: int = 0,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into train, validation and test sets.

    Args:
        data_df: The data frame.
        cat_columns: The names of the categorical columns.
        num_train: The number of samples to be used for training.
        num_test: The number of samples to be used for testing.
        random_seed: The random seed to be used for the split.

    Returns:
        A tuple of the train dataframe and the test dataframe.
    """
    total_num = data_df.shape[0]
    idx = np.arange(total_num)

    while True:
        np.random.seed(random_seed)
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
        random_seed += 1

    np.random.seed(None)
    return train_df, test_df


# TODO: Refactor this function to get rid of the "too many statements" and "too many branches" errors from Ruff
def process_data(name: str, info_path: Path, data_dir: Path) -> None:
    # ruff: noqa: PLR0915, PLR0912
    """Process the data for the given dataset name.

    It will save the processed data to the processed data directory.

    Args:
        name: The name of the dataset.
        info_path: The directory where the info file is located.
        data_dir: The directory where the raw data is located.
    """
    processed_data_dir = get_processed_data_dir(data_dir)

    with open(info_path / f"{name}.json", "r") as f:
        info = json.load(f)

    data_path = info["data_path"]
    if info["file_type"] == "csv":
        data_df = pd.read_csv(data_path, header=info["header"])

    elif info["file_type"] == "xls":
        data_df = pd.read_excel(data_path, sheet_name="Data", header=1)
        data_df = data_df.drop("ID", axis=1)

    else:
        raise ValueError(f"Unsupported file type: {info['file_type']}. Supported file types are: ['csv', 'xls'].")

    num_rows = data_df.shape[0]

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

    if info["test_path"]:
        # if testing data is given
        test_path = info["test_path"]
        test_df = pd.read_csv(test_path)
        train_df = data_df
    else:
        # Train/ Test Split, 99% Training, 1% Testing (Validation set will be selected from Training set)
        num_train = int(num_rows * 0.99)
        num_test = num_rows - num_train

        train_df, test_df = train_val_test_split(data_df, cat_columns, num_train, num_test)

    train_df.columns = list(range(len(train_df.columns)))
    test_df.columns = list(range(len(test_df.columns)))

    # TODO: this is never used. Remove this block of code to avoid confusion.
    col_info: dict[int | str, Any] = {}

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
    test_df.rename(columns=idx_name_mapping, inplace=True)

    for col in num_columns:
        train_df.loc[train_df[col] == "?", col] = np.nan
    for col in cat_columns:
        if pd.api.types.is_string_dtype(train_df[col]) or pd.api.types.is_object_dtype(train_df[col]):
            train_df.loc[train_df[col] == "?", col] = "nan"
    for col in num_columns:
        test_df.loc[test_df[col] == "?", col] = np.nan
    for col in cat_columns:
        if pd.api.types.is_string_dtype(test_df[col]) or pd.api.types.is_object_dtype(test_df[col]):
            test_df.loc[test_df[col] == "?", col] = "nan"

    numerical_features_train = train_df[num_columns].to_numpy().astype(np.float32)
    categorical_features_train = train_df[cat_columns].to_numpy().astype(np.int64)
    target_train = train_df[target_columns].to_numpy()

    numerical_features_test = test_df[num_columns].to_numpy().astype(np.float32)
    categorical_features_test = test_df[cat_columns].to_numpy().astype(np.int64)
    target_test = test_df[target_columns].to_numpy()

    if not (processed_data_dir / name).exists():
        (processed_data_dir / name).mkdir(parents=True)

    np.save(processed_data_dir / name / "x_num_train.npy", numerical_features_train)
    np.save(processed_data_dir / name / "x_cat_train.npy", categorical_features_train)
    np.save(processed_data_dir / name / "y_train.npy", target_train)

    np.save(processed_data_dir / name / "x_num_test.npy", numerical_features_test)
    np.save(processed_data_dir / name / "x_cat_test.npy", categorical_features_test)
    np.save(processed_data_dir / name / "y_test.npy", target_test)

    train_df[num_columns] = train_df[num_columns].astype(np.float32)
    test_df[num_columns] = test_df[num_columns].astype(np.float32)

    train_df.to_csv(processed_data_dir / name / "train.csv", index=False)
    test_df.to_csv(processed_data_dir / name / "test.csv", index=False)

    info["column_names"] = column_names
    info["train_num"] = train_df.shape[0]
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

    if task_type == TaskType.REGRESSION.value:
        for i in target_col_idx:
            metadata["columns"][i] = {}
            metadata["columns"][i]["sdtype"] = "numerical"
            metadata["columns"][i]["computer_representation"] = "Float"

    else:
        for i in target_col_idx:
            metadata["columns"][i] = {}
            metadata["columns"][i]["sdtype"] = "categorical"

    info["metadata"] = metadata

    with open(processed_data_dir / name / "info.json", "w") as file:
        json.dump(info, file, indent=4)

    log(INFO, f"Processing and Saving {name} Successfully!")

    log(INFO, f"Dataset Name: {name}")
    log(INFO, f"Total Size: {info['train_num'] + info['test_num']}")
    log(INFO, f"Train Size: {info['train_num']}")
    log(INFO, f"Test Size: {info['test_num']}")
    if info["task_type"] == TaskType.REGRESSION.value:
        num = len(info["num_col_idx"] + info["target_col_idx"])
        cat = len(info["cat_col_idx"])
    else:
        cat = len(info["cat_col_idx"] + info["target_col_idx"])
        num = len(info["num_col_idx"])
    log(INFO, f"Number of Numerical Columns: {num}")
    log(INFO, f"Number of Categorical Columns: {cat}")
