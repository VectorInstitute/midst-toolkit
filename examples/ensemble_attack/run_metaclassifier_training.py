import pickle
from logging import INFO
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from midst_toolkit.attacks.ensemble.blending import BlendingPlusPlus, MetaClassifierType
from midst_toolkit.attacks.ensemble.data_utils import load_dataframe
from midst_toolkit.common.logger import log


def run_metaclassifier_training(
    config: DictConfig,
    shadow_data_paths: list[Path],
    target_model_synthetic_path: Path,
) -> None:
    """
    Function to run the metaclassifier training and evaluation.

    Args:
        config: Configuration object set in config.yaml.
        shadow_data_paths: List of paths to the trained shadow models and all their attributes and synthetic data.
            The list should contain three paths, one for each set of shadow models.
        target_model_synthetic_path: Path to the target model's synthetic data. This is all we need from a target
            model to train the metaclassifier in the black-box setting.
        target_model_synthetic_path: Path to the target model's synthetic data. This is all we need from a target
            model to train the metaclassifier in the black-box setting.
    """
    log(INFO, "Running metaclassifier training...")

    # Load the processed data splits.
    df_meta_train = load_dataframe(
        Path(config.data_paths.processed_attack_data_path),
        "master_challenge_train.csv",
    )

    # y_meta_train consists of binary labels (0s and 1s) indicating whether each row in df_meta_train
    # belongs to the target model's training set.
    y_meta_train = np.load(
        Path(config.data_paths.processed_attack_data_path) / "master_challenge_train_labels.npy",
    )
    df_meta_test = load_dataframe(
        Path(config.data_paths.processed_attack_data_path),
        "master_challenge_test.csv",
    )
    y_meta_test = np.load(
        Path(config.data_paths.processed_attack_data_path) / "master_challenge_test_labels.npy",
    )

    # Three sets of shadow models are trained separately and their paths are provided here.

    assert len(shadow_data_paths) == 3, (
        "At this point of development, the shadow_data_paths list must contain exactly three elements."
    )

    shadow_data_collection = []

    for model_path in shadow_data_paths:
        assert model_path.exists(), (
            f"No file found at {model_path}. Make sure the path is correct, or run shadow model training first."
        )

        with open(model_path, "rb") as f:
            shadow_data_and_result = pickle.load(f)
            shadow_data_collection.append(shadow_data_and_result)
            log(INFO, f"Shadow model data loaded from {model_path}.")

    assert Path(target_model_synthetic_path).exists(), (
        f"No file found at {target_model_synthetic_path}. "
        f"Make sure the path is correct and that you have access to target model's synthetic data."
    )

    # Load the target model's synthetic data
    target_synthetic_data = pd.read_csv(target_model_synthetic_path)
    log(
        INFO,
        f"Target model's synthetic data loaded from {target_model_synthetic_path} with size {len(target_synthetic_data)}.",
    )

    assert target_synthetic_data is not None, "Target model's synthetic data is missing."
    target_synthetic_data = target_synthetic_data.copy()

    df_reference = load_dataframe(
        Path(config.data_paths.population_path),
        "population_all_with_challenge_no_id.csv",
    )
    log(
        INFO,
        f"Reference population data loaded from f{config.data_paths.population_path} with size {len(df_reference)}.",
    )

    # Extract trans_id from both train and test dataframes
    assert "trans_id" in df_meta_train.columns, "Meta train data must have trans_id column"
    train_trans_ids = df_meta_train["trans_id"]

    assert "trans_id" in df_meta_test.columns, "Meta test data must have trans_id column"
    test_trans_ids = df_meta_test["trans_id"]

    df_meta_train = df_meta_train.drop(columns=["trans_id", "account_id"])
    df_meta_test = df_meta_test.drop(columns=["trans_id", "account_id"])

    # Fit the metaclassifier.
    meta_classifier_type = MetaClassifierType(config.metaclassifier.model_type)

    # 1. Initialize the attacker
    blending_attacker = BlendingPlusPlus(
        config=config,
        shadow_data_collection=shadow_data_collection,
        data_types_file_path=Path(config.metaclassifier.data_types_file_path),
        meta_classifier_type=meta_classifier_type,
        random_seed=config.random_seed,
    )

    log(INFO, f"{meta_classifier_type} created with random seed {config.random_seed}.")

    # 2. Train the attacker on the meta-train set
    blending_attacker.fit(
        df_train=df_meta_train,
        y_train=y_meta_train,
        df_target_synthetic=target_synthetic_data,
        df_reference=df_reference,
        id_column_data=train_trans_ids,
        use_gpu=config.metaclassifier.use_gpu,
        epochs=config.metaclassifier.epochs,
    )

    model_filename = config.metaclassifier.meta_classifier_model_name
    model_path = Path(config.model_paths.metaclassifier_model_path) / f"{model_filename}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(blending_attacker.trained_model, f)

    log(INFO, "Metaclassifier model saved, starting evaluation...")

    # 3. Get predictions on the meta test set (evaluation of the trained metaclassifier)
    # For evaluation, we test the meta classifier on the meta test set provided the target's synthetic data.
    probabilities, pred_score = blending_attacker.predict(
        df_test=df_meta_test,
        df_original_synthetic=target_synthetic_data,  # For evaluation only
        df_reference=df_reference,
        id_column_data=test_trans_ids,
        y_test=y_meta_test,
    )

    # Save the evaluation prediction probabilities
    attack_evaluation_result_path = Path(config.data_paths.attack_evaluation_result_path)
    attack_evaluation_result_path.mkdir(parents=True, exist_ok=True)
    file_name = attack_evaluation_result_path / f"{model_filename}_val_pred_proba.npy"
    np.save(file_name, probabilities)
    log(INFO, f"Evaluation prediction probabilities saved at {file_name}.")

    if pred_score is not None:
        log(INFO, f"TPR at FPR=0.1: {pred_score:.4f}")
