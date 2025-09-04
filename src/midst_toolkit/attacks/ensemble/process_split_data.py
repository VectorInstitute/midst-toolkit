from logging import INFO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from midst_toolkit.attacks.ensemble.data_utils import save_dataframe
from midst_toolkit.common.logger import log


def split_real_data(
    df_real: pd.DataFrame,
    column_to_stratify: str | None = None,
    proportion: dict | None = None,
    random_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a real dataset into train, validation, and test sets, saves them as CSV files, and returns the splits.

    Args:
        df_real: The input real dataset to be split.
        column_to_stratify: Column name to use for stratified splitting.
        proportion: Proportions for train and validation splits.
        random_seed: Random seed for reproducibility.

    Returns:
        A tuple containing the train, validation, and test dataframes.
    """
    if proportion is None:
        proportion = {"train": 0.5, "val": 0.25}

    # Split the real data into train and control
    df_real_train, df_real_control = train_test_split(
        df_real,
        test_size=1 - proportion["train"],
        random_state=random_seed,
        stratify=df_real[column_to_stratify],
    )

    # Further split the control into val and test set:
    df_real_val, df_real_test = train_test_split(
        df_real_control,
        test_size=(1 - proportion["train"] - proportion["val"]) / (1 - proportion["train"]),
        random_state=random_seed,
        stratify=df_real_control[column_to_stratify],
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
    random_seed: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Generates the validation and test sets with labels.
    The resulting validation and test sets are used for meta classifier training and evaluation, respectively.

    Args:
        df_real_train: Real training data.
        df_real_control_val: Real control data for validation.
        df_real_control_test: Real control data for final evaluation.
        stratify: Series used to stratify the real training data.
        random_seed: Random seed for reproducibility.

    Returns:
        Features and labels for validation and test sets, respectively.
    """
    df_real_train["stratify"] = stratify

    # Construct validation set for ensemble model
    df_real_train_val, df_temp = train_test_split(
        df_real_train,
        train_size=len(df_real_control_val),
        stratify=df_real_train["stratify"],
        random_state=random_seed,
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
    df_val = df_val.sample(frac=1, random_state=random_seed).reset_index(drop=True)

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
            random_state=random_seed,
        )

    df_real_train_test["is_train"] = 1
    df_real_control_test["is_train"] = 0

    df_test = pd.concat(
        [df_real_train_test.drop(columns=["stratify"]), df_real_control_test],
        axis=0,
        ignore_index=True,
    )

    df_test = df_test.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    y_test = df_test["is_train"].values
    df_test = df_test.drop(columns=["is_train"])

    return df_val, y_val, df_test, y_test


def process_split_data(
    all_population_data: pd.DataFrame,
    processed_attack_data_path: Path,
    column_to_stratify: str,
    num_total_samples: int = 40000,
    random_seed: int = 42,
) -> None:
    """
    Splits the data into train, validation, and test sets according to the attack design.

    Args:
        all_population_data: The total population data that the attacker has access to as a DataFrame.
        processed_attack_data_path: Path where the processed attack data will be saved.
        column_to_stratify: Column name to use for stratified splitting.
        num_total_samples: Number os samples that I randomly selected from the population. Defaults to 40000.
        random_seed: Random seed used for reproducibility. Defaults to 42.
    """
    # Original Ensemble attack samples 40k data points to construct
    # 1) the main population (real data) used for training the synthetic data generator model,
    # 2) evaluation that is the meta train data used to train the meta classifier,
    # 3) test to evaluate the meta classifier.

    df_real_data = all_population_data.sample(n=num_total_samples, random_state=random_seed)

    # Split the data. df_real_train is used for training the synthetic data generator model.
    df_real_train, df_real_val, df_real_test = split_real_data(
        df_real_data,
        column_to_stratify=column_to_stratify,
        random_seed=random_seed,
    )
    # Generate validation and test sets with labels. Validation is used for training the meta classifier
    # and test is used for meta classifier evaluation.
    # Half of the df_real_train will be assigned to validation and the other half to test with
    # their "is_train" column set to 1 meaning that these samples are in the models training corpus.
    df_val, y_val, df_test, y_test = generate_val_test(
        df_real_train,
        df_real_val,
        df_real_test,
        stratify=df_real_train[column_to_stratify],  # TODO: This value is not documented in the original codebase.
        random_seed=random_seed,
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
    log(INFO, f"Data splits saved to {processed_attack_data_path}")
