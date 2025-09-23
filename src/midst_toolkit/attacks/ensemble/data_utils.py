import json
import os
from logging import INFO, WARNING
from pathlib import Path

import pandas as pd

from midst_toolkit.common.logger import log
from midst_toolkit.core.data_loaders import get_info_from_domain, pipeline_process_data


def save_dataframe(df: pd.DataFrame, file_path: Path, file_name: str) -> None:
    """
    Save a DataFrame to a CSV file.

    Args:
        df: DataFrame to be saved.
        file_path: Path where the file will be saved.
        file_name: Name of the file to save the DataFrame as.

    Returns:
        None
    """
    assert Path.exists(file_path), f"Path {file_path} does not exist."
    if Path.exists(file_path / file_name):
        log(
            WARNING,
            f"File {file_path / file_name} already exists and will be overwritten.",
        )
    df.to_csv(file_path / file_name, index=False)
    log(INFO, f"DataFrame saved to {file_path / file_name}")


def load_dataframe(file_path: Path, file_name: str) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV file.

    Args:
        file_path: Path where the file is stored.
        file_name: Name of the file to load the DataFrame from.

    Returns:
        Loaded dataframe.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    full_path = Path(file_path / file_name)
    if not Path.exists(full_path):
        raise FileNotFoundError(f"File {full_path} does not exist.")
    # Assert that the file is a CSV file
    assert full_path.suffix == ".csv", f"File {file_name} is not a CSV file."
    df = pd.read_csv(full_path)
    log(INFO, f"DataFrame loaded from {full_path}")
    return df


# TODO: Merge with original. The following function is the slightly modified version of
# ``midst_toolkit.core.data_loaders.load_multi_table`` by the CITADEL & UQAM team.
def load_multi_table(data_dir, train_df=None, verbose=True):
    dataset_meta = json.load(open(os.path.join(data_dir, "dataset_meta.json"), "r"))

    relation_order = dataset_meta["relation_order"]
    relation_order_reversed = relation_order[::-1]

    tables = {}

    for table, meta in dataset_meta["tables"].items():
        if train_df is None:
            if os.path.exists(os.path.join(data_dir, "train.csv")):
                train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
            else:
                train_df = pd.read_csv(os.path.join(data_dir, f"{table}.csv"))
        tables[table] = {
            "df": train_df,
            "domain": json.load(open(os.path.join(data_dir, f"{table}_domain.json"))),
            "children": meta["children"],
            "parents": meta["parents"],
        }
        tables[table]["original_cols"] = list(tables[table]["df"].columns)
        tables[table]["original_df"] = tables[table]["df"].copy()
        id_cols = [col for col in tables[table]["df"].columns if "_id" in col]
        df_no_id = tables[table]["df"].drop(columns=id_cols)
        info = get_info_from_domain(df_no_id, tables[table]["domain"])
        data, info = pipeline_process_data(
            name=table,
            data_df=df_no_id,
            info=info,
            ratio=1,
            save=False,
            verbose=verbose,
        )
        tables[table]["info"] = info

    return tables, relation_order, dataset_meta


# TODO: the following function is directly copied from the midst reference code since
# I need it to run the code, but, it should be moved to somewhere else.
def load_configs(config_path):
    configs = json.load(open(config_path, "r"))

    save_dir = os.path.join(configs["general"]["workspace_dir"], configs["general"]["exp_name"])
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "before_matching"), exist_ok=True)

    with open(os.path.join(save_dir, "args"), "w") as file:
        json.dump(configs, file, indent=4)

    return configs, Path(save_dir)
