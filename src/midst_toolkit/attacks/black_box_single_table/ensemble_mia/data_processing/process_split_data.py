from logging import INFO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from midst_toolkit.common.logger import log
from midst_toolkit.attacks.black_box_single_table.ensemble_mia.config import (
    seed,
)
from midst_toolkit.attacks.black_box_single_table.ensemble_mia.data_processing.data_utils import (
    save_dataframe,
)
from midst_toolkit.attacks.black_box_single_table.ensemble_mia.data_processing.real_data_collection import (
    collect_population_data_ensemble_mia,
)


def split_real_data(
    df_real: pd.DataFrame,
    var_to_stratify: str | None = None,
    proportion: dict | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a real dataset into train, validation, and test sets, saves them as CSV files, and returns the splits.

    Args:
        df_real (pd.DataFrame): The input real dataset to be split.
        var_to_stratify (str, optional): Column name to use for stratified splitting. Defaults to None.
        proportion (dict, optional): Proportions for train and validation splits.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the train, validation, and test DataFrames.
    """
    if proportion is None:
        proportion = {"train": 0.5, "val": 0.25}

    # Split the real data into train and control
    df_real_train, df_real_control = train_test_split(
        df_real,
        test_size=1 - proportion["train"],
        random_state=seed,
        stratify=df_real[var_to_stratify],
    )

    # Further split the control into val and test set:
    df_real_val, df_real_test = train_test_split(
        df_real_control,
        test_size=(1 - proportion["train"] - proportion["val"]) / (1 - proportion["train"]),
        random_state=seed,
        stratify=df_real_control[var_to_stratify],
    )

    return (
        df_real_train,
        df_real_val,
        df_real_test,
    )


def generate_val_test(
    df_real_train: pd.DataFrame,
    df_real_control_val: pd.DataFrame,
    df_real_control_test: pd.DataFrame,
    stratify: pd.Series,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Generates the validation and test sets with labels. 
    The resulting validation and test sets are used for meta classifier training and evaluation, respectively.

    Args:
        df_real_train (pd.DataFrame): Real training data.
        df_real_control_val (pd.DataFrame): Real control data for validation.
        df_real_control_test (pd.DataFrame): Real control data for final evaluation.
        stratify (pd.Series): Series used to stratify the real training data.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]: Features and labels for validation and test sets.
    """
    df_real_train["stratify"] = stratify

    # Construct validation set for ensemble model
    df_real_train_val, df_temp = train_test_split(
        df_real_train,
        train_size=len(df_real_control_val),
        stratify=df_real_train["stratify"],
        random_state=seed,
    )

    #  Label 1 for real records used to generate 1st generation synthetic data and 0 for control
    df_real_train_val["is_train"] = 1
    df_real_control_val["is_train"] = 0

    df_val = pd.concat(
        [df_real_train_val.drop(columns=["stratify"]), df_real_control_val],
        axis=0,
        ignore_index=True,
    )

    # Shuffle data
    df_val = df_val.sample(frac=1, random_state=seed).reset_index(drop=True)

    y_val = df_val["is_train"].values
    df_val = df_val.drop(columns=["is_train"])

    # Test set: can be used to evaluate all the models
    if len(df_temp) == len(df_real_control_test):
        df_real_train_test = df_temp
    else:
        df_real_train_test, _ = train_test_split(
            df_temp,
            train_size=len(df_real_control_test),
            stratify=df_temp["stratify"],
            random_state=seed,
        )

    df_real_train_test["is_train"] = 1
    df_real_control_test["is_train"] = 0

    df_test = pd.concat(
        [df_real_train_test.drop(columns=["stratify"]), df_real_control_test],
        axis=0,
        ignore_index=True,
    )

    df_test = df_test.sample(frac=1, random_state=seed).reset_index(drop=True)

    y_test = df_test["is_train"].values
    df_test = df_test.drop(columns=["is_train"])

    return df_val, y_val, df_test, y_test


def process_split_data(
    data_config: dict, population_data_file_name: str = "population_all_with_challenge.csv"
) -> None:
    """
    Calls `real_data_collection.collect_population_data_ensemble_mia` to collect the population data
    as specified by the `population_data_file_name` argument, then splits the data into train,
    validation, and test sets according to the attack design.
    """
    # Input path
    population_path = data_config["population_path"]
    # output_path
    processed_attack_data_path = data_config["processed_attack_data_path"]

    # Check if the input file exists, if not create it.
    if (population_path / population_data_file_name).exists():
        log(
            INFO,
            f"Population data {population_data_file_name} already exists. Skipping collection.",
        )
    else:
        _ = collect_population_data_ensemble_mia(
            data_config=data_config,
        )
    all_population_data = pd.read_csv(population_path / population_data_file_name)

    # Sample 40k data points to construct the main population (real data) used for training the
    # synthetic data generator model, evaluation (meta train data used to train the meta classifier),
    # and test (to evaluate the meta classifier).

    df_real_data = all_population_data.sample(n=40000, random_state=seed)

    # Split the data. df_real_train is used for training the synthetic data generator model.
    df_real_train, df_real_val, df_real_test = split_real_data(
        df_real_data,
        var_to_stratify="trans_type",  # TODO: This value is not documented in the original codebase.
        seed=seed,
    )
    # Generate validation and test sets with labels. Validation is used for training the meta classifier
    # and test is used for meta classifier evaluation.
    # Half of the df_real_train will be assigned to validation and the other half to test with
    # their "is_train" column set to 1 meaning that these samples are in the models training corpus.
    df_val, y_val, df_test, y_test = generate_val_test(
        df_real_train,
        df_real_val,
        df_real_test,
        stratify=df_real_train["trans_type"],  # TODO: This value is not documented in the original codebase.
        seed=seed,
    )

    df_real_train = df_real_train.drop(columns=["stratify"])
    df_real_val = df_real_val.drop(columns=["is_train"])
    df_real_test = df_real_test.drop(columns=["is_train"])

    save_dataframe(df_real_train, processed_attack_data_path, "real_train.csv")
    save_dataframe(df_real_val, processed_attack_data_path, "real_val.csv")
    save_dataframe(df_real_test, processed_attack_data_path, "real_test.csv")

    save_dataframe(df_val, processed_attack_data_path, "master_challenge_train.csv")
    np.save(
        processed_attack_data_path / "master_challenge_train_labels.npy",
        y_val,
    )
    save_dataframe(df_test, processed_attack_data_path, "master_challenge_test.csv")
    np.save(
        processed_attack_data_path / "master_challenge_test_labels.npy",
        y_test,
    )


if __name__ == "__main__":
    from midst_toolkit.attacks.black_box_single_table.ensemble_mia.config import (
        DATA_CONFIG,
    )
    process_split_data(data_config=DATA_CONFIG)
