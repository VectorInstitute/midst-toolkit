import pickle
from datetime import datetime
from logging import INFO
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from midst_toolkit.attacks.ensemble.blending import BlendingPlusPlus, MetaClassifierType
from midst_toolkit.attacks.ensemble.data_utils import load_dataframe
from midst_toolkit.common.logger import log


def run_metaclassifier_training(config: DictConfig, attack_data_paths: list[str], target_data_path: str) -> None:
    """
    Fuction to run the metaclassifier training and evaluation.

    Args:
        config: Configuration object set in config.yaml.
        attack_data_paths: List of paths to the trained shadow models and all their attributes and synthetic data.
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
    base_path = Path(config.model_paths.shadow_models_path)
    attack_data_collection = []
    for model_path in attack_data_paths:
        final_model_path = base_path / model_path
        with open(final_model_path, "rb") as f:
            shadow_model = pickle.load(f)
            attack_data_collection.append(shadow_model)

    # TODO: Uncomment after we get a target model.
    target_data_collection = []
    # with open(target_data_path, "rb") as f:
    #     target_model = pickle.load(f)
    #     target_data_collection.append(target_model)

    # import pdb; pdb.set_trace()

    dummy_target_set = attack_data_collection[1]["fine_tuning_sets"][2]
    dummy_target_results = attack_data_collection[1]["fine_tuned_results"][2]
    # TODO: Do we need a list of target models or just one is enough? (Depends on RMIA functions structure)
    target_data_collection.append(
        {
            "fine_tuning_sets": [dummy_target_set],
            "fine_tuned_results": [dummy_target_results],
        }
    )

    # Synthetic data borrowed from the attack implementation repository.
    # From (https://github.com/CRCHUM-CITADEL/ensemble-mia/tree/main/input/tabddpm_black_box/meta_classifier)
    # TODO: Change this file path to the path where the synthetic data is stored, or get from the target model.
    df_synthetic = load_dataframe(
        Path(config.data_paths.processed_attack_data_path),
        "synth.csv",
    )

    df_reference = load_dataframe(
        Path(config.data_paths.population_path),
        "population_all_with_challenge_no_id.csv",
    )

    # Extract trans_id from both train and test dataframes
    if "trans_id" in df_meta_train.columns:
        train_trans_ids = df_meta_train["trans_id"]
    else:
        raise Exception("Train data must have trans_id column")

    if "trans_id" in df_meta_test.columns:
        train_trans_ids = df_meta_train["trans_id"]
    else:
        raise Exception("Test data must have trans_id column")

    # We should drop the id column from master metaclassifier train data.
    if "trans_id" in df_meta_train.columns:
        df_meta_train = df_meta_train.drop(columns=["trans_id", "account_id"])
    if "trans_id" in df_meta_test.columns:
        df_meta_test = df_meta_test.drop(columns=["trans_id", "account_id"])

    # Fit the metaclassifier.
    meta_classifier_enum = MetaClassifierType(config.metaclassifier.model_type)

    # 1. Initialize the attacker
    blending_attacker = BlendingPlusPlus(
        config=config,
        attack_data_collection=attack_data_collection,
        target_data_collection=target_data_collection,
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

    log(INFO, "Metaclassifier training finished.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{timestamp}_{config.metaclassifier.model_type}_trained_metaclassifier.pkl"
    with open(Path(config.model_paths.metaclassifier_model_path) / model_filename, "wb") as f:
        pickle.dump(blending_attacker.trained_model, f)

    log(INFO, "Metaclassifier model saved, starting evaluation...")

    # 3. Get predictions on the test set
    probabilities, pred_score = blending_attacker.predict(
        df_test=df_meta_test,
        df_synthetic=df_synthetic,
        df_reference=df_reference,
        y_test=y_meta_test,
    )

    # Save the prediction probabilities
    np.save(
        Path(config.data_paths.attack_results_path)
        / f"{timestamp}_{config.metaclassifier.model_type}_test_pred_proba.npy",
        probabilities,
    )
    log(INFO, "Test set prediction probabilities saved.")

    if pred_score is not None:
        log(INFO, f"TPR at FPR=0.1: {pred_score:.4f}")
