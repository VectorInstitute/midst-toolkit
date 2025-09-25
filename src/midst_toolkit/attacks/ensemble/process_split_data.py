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
    proportion: dict[str, float] | None = None,
    random_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a real dataset into train, validation, and test sets, saves them as CSV files, and returns the splits.

    Args:
        df_real: The input real dataset to be split.
        column_to_stratify: Column name to use for stratified splitting. If provided, the function
            ensures that the distribution of values in this column is preserved across the splits.
            If None, no stratification is applied. Defaults to None.
        proportion: Proportions for train and validation splits. If None, defaults to {"train": 0.5, "val": 0.25}.
            The test set proportion will be inferred as 1 - (train + val). Defaults to None.
        random_seed: Random seed for reproducibility. If None, you might get different splits each time.
            Defaults to None.

    Returns:
        A tuple containing the train, validation, and test dataframes.
    """
    if proportion is None:
        proportion = {"train": 0.5, "val": 0.25}
    else:
        # Sanity check for proportion values
        assert "train" in proportion and "val" in proportion, "Proportion must contain 'train' and 'val' keys."
        assert 0 < proportion["train"] < 1, "Train proportion must be between 0 and 1."
        assert 0 < proportion["val"] < 1, "Validation proportion must be between 0 and 1."
        assert proportion["train"] + proportion["val"] < 1, (
            "Sum of train and validation proportions must be less than 1."
        )

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


def generate_train_test_challenge_splits(
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
        stratify: Series used to stratify the real training data. This column is added to read train data
            as "stratify" and is used for stratified splitting. This ensures that the distribution of values
            in this column is preserved across the splits.
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

    y_val = df_val["is_train"].to_numpy()
    df_val = df_val.drop(columns=["is_train"])

    # Test set
    # `df_temp` will be assigned as our test set if it has the same size as `df_real_control_test`,
    # otherwise, we further split `df_temp` to get a test set of the same size as `df_real_control_test`.
    # This is because we want to take a train split of same size as `df_real_control_test` to ensure
    # balanced classes in the final test set.
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

    y_test = df_test["is_train"].to_numpy()
    df_test = df_test.drop(columns=["is_train"])

    return df_val, y_val, df_test, y_test


def process_split_data(
    all_population_data: pd.DataFrame,
    processed_attack_data_path: Path,
    column_to_stratify: str,
    num_total_samples: int = 40000,
    challenge_data_size: int = 10000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into train, validation, and test sets according to the attack design.

    Args:
        all_population_data: The total population data that the attacker has access to as a DataFrame.
        processed_attack_data_path: Path where the processed attack data will be saved.
        column_to_stratify: Column name to use for stratified splitting.
        num_total_samples: Number os samples that are randomly selected from the population. Defaults to 40000.
        random_seed: Random seed used for reproducibility. Defaults to 42.

    Returns:
        A tuple containing the train, validation, and test dataframes for real data,
        as well as the validation and test dataframes for the challenge dataset.

    """
    # Original Ensemble attack samples 40k data points to construct
    # 1) the main population (real data) used for training the synthetic data generator model,
    # 2) evaluation that is the meta train data (membership classification train dataset) used to train
    #    the meta classifier,
    # 3) test (membership classification test dataset) to evaluate the meta classifier.

    df_real_data = all_population_data.sample(n=num_total_samples, random_state=random_seed)

    # `df_real_train` is used for training the synthetic data generator model.
    df_real_train, df_real_val, df_real_test = split_real_data(
        df_real_data,
        column_to_stratify=column_to_stratify,
        random_seed=random_seed,
    )
    # Generate challenge datasets:
    # `df_val` is used for training the meta classifier (membership classification train dataset).
    # and `df_test` is used for meta classifier evaluation (membership classification test dataset).
    # A part of the `df_real_train` will be assigned to `df_val` and a another part to `df_test` with
    # their "is_train" column set to 1 meaning that these samples are in the models training corpus.
    # Because `df_real_train` will be used to train a synthetic model, we're including some of it in
    # `df_val` and `df_test` sets to create the challenges assuming the `df_real_val` and `df_real_test`
    # data will not be part of the training data.
    # This code makes sure `is_train` classes are balanced in the challenge datasets.
    df_val, y_val, df_test, y_test = generate_train_test_challenge_splits(
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

    return df_real_train, df_real_val, df_real_test, df_val, df_test
