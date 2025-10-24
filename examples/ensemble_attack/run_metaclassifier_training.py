import pickle
from datetime import datetime
from logging import INFO
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from midst_toolkit.attacks.ensemble.blending import BlendingPlusPlus, MetaClassifierType
from midst_toolkit.attacks.ensemble.data_utils import load_dataframe
from midst_toolkit.common.logger import log


def run_metaclassifier_training(
    config: DictConfig,
    attack_data_paths: list[Path],
    target_data_path: Path,
) -> None:
    """
    Fuction to run the metaclassifier training and evaluation.

    Args:
        config: Configuration object set in config.yaml.
        attack_data_paths: List of paths to the trained shadow models and all their attributes and synthetic data.
            The list should contain three paths, one for each set of shadow models.
        target_data_path: Path to the target model and all its attributes and synthetic data.
    """
    log(INFO, "Running metaclassifier training...")

    # Load the processed data splits.
    df_meta_train = load_dataframe(
        Path(config.data_paths.processed_attack_data_path),
        "master_challenge_train.csv",
    )
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

    assert len(attack_data_paths) == 3, (
        "At this point of development, the attack_data_paths list must contain exactly three elements."
    )

    attack_data_collection = []

    for model_path in attack_data_paths:
        assert model_path.exists(), (
            f"No file found at {model_path}. Make sure the path is correct, or run shadow model training first."
        )

        with open(model_path, "rb") as f:
            shadow_data_and_result = pickle.load(f)
            attack_data_collection.append(shadow_data_and_result)

    assert target_data_path.exists(), (
        f"No file found at {target_data_path}. Make sure the path is correct and that you have trained the target model."
    )

    with open(target_data_path, "rb") as f:
        target_data_and_result = pickle.load(f)

    synth = target_data_and_result["trained_results"][0].synthetic_data
    assert synth is not None, "Target model pickle missing synthetic_data."
    df_synthetic = synth.copy()

    df_reference = load_dataframe(
        Path(config.data_paths.population_path),
        "population_all_with_challenge_no_id.csv",
    )

    # Extract trans_id from both train and test dataframes
    assert "trans_id" in df_meta_train.columns, "Meta train data must have trans_id column"
    train_trans_ids = df_meta_train["trans_id"]

    assert "trans_id" in df_meta_test.columns, "Meta test data must have trans_id column"
    test_trans_ids = df_meta_test["trans_id"]

    df_meta_train = df_meta_train.drop(columns=["trans_id", "account_id"])
    df_meta_test = df_meta_test.drop(columns=["trans_id", "account_id"])

    # Fit the metaclassifier.
    meta_classifier_enum = MetaClassifierType(config.metaclassifier.model_type)

    # 1. Initialize the attacker
    blending_attacker = BlendingPlusPlus(
        config=config,
        attack_data_collection=attack_data_collection,
        target_data=target_data_and_result,
        meta_classifier_type=meta_classifier_enum,
        random_seed=config.random_seed,
    )

    log(INFO, f"{meta_classifier_enum} created with random seed {config.random_seed}.")

    # 2. Train the attacker on the meta-train set

    blending_attacker.fit(
        df_train=df_meta_train,
        y_train=y_meta_train,
        df_synthetic=df_synthetic,
        df_reference=df_reference,
        id_column_data=train_trans_ids,
        use_gpu=config.metaclassifier.use_gpu,
        epochs=config.metaclassifier.epochs,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{timestamp}_{config.metaclassifier.model_type}_trained_metaclassifier.pkl"
    with open(Path(config.model_paths.metaclassifier_model_path) / model_filename, "wb") as f:
        pickle.dump(blending_attacker.trained_model, f)

    log(INFO, "Metaclassifier model saved, starting evaluation...")

    # Get the synthetic data provided by the challenge for evaluation
    # TODO: Check if the file is the correct one.
    df_synthetic_original = load_dataframe(
        Path(config.data_paths.processed_attack_data_path),
        "synth.csv",
    )

    # 3. Get predictions on the test set
    probabilities, pred_score = blending_attacker.predict(
        df_test=df_meta_test,
        df_synthetic=df_synthetic_original,
        df_reference=df_reference,
        id_column_data=test_trans_ids,
        y_test=y_meta_test,
    )

    # Save the prediction probabilities
    attack_results_path = Path(config.data_paths.attack_results_path)
    attack_results_path.mkdir(parents=True, exist_ok=True)
    np.save(
        Path(config.data_paths.attack_results_path)
        / f"{timestamp}_{config.metaclassifier.model_type}_test_pred_proba.npy",
        probabilities,
    )
    log(INFO, "Test set prediction probabilities saved.")

    if pred_score is not None:
        log(INFO, f"TPR at FPR=0.1: {pred_score:.4f}")
