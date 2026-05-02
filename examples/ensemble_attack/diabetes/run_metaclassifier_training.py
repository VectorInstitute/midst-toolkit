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
    # shadow_data_paths: list[Path],
    # target_model_synthetic_path: Path,
) -> None:
    """
    Function to run the metaclassifier training and evaluation.

    Args:
        config: Configuration object set in config.yaml.
        shadow_data_paths: List of paths to the trained shadow models and all their attributes and synthetic data.
            The list should contain three paths, one for each set of shadow models.
        target_model_synthetic_path: Path to the target model's synthetic data. This is all we need from a target
            model to train the metaclassifier in the black-box setting.
    """
    # log(INFO, "Running metaclassifier training...")

    # # Load the processed data splits.

    # df_meta_train_1 = load_dataframe(
    #     Path("/projects/midst-experiments/ensemble_attack/number_of_target/42/attack_data"), "master_challenge_train.csv"
    #     )
    # df_meta_train_2 = load_dataframe(
    #     Path("/projects/midst-experiments/ensemble_attack/number_of_target/44/attack_data"), "master_challenge_train.csv"
    #     )

    # df_meta_train = pd.concat([df_meta_train_1, df_meta_train_2], ignore_index=True)

    # y_meta_train consists of binary labels (0s and 1s) indicating whether each row in df_meta_train
    # belongs to the target model's training set.

    
    y_meta_train_1 = np.load(
        "/projects/midst-experiments/ensemble_attack/diabetes_experiments/number_of_targets/42/attack_data/master_challenge_train_labels.npy"
    )

    y_meta_train_2 = np.load(
        "/projects/midst-experiments/ensemble_attack/diabetes_experiments/number_of_targets/43/attack_data/master_challenge_train_labels.npy"

    )

    y_meta_train_3 = np.load(
        "/projects/midst-experiments/ensemble_attack/diabetes_experiments/number_of_targets/44/attack_data/master_challenge_train_labels.npy"

    )

    y_meta_train_4 = np.load(
        "/projects/midst-experiments/ensemble_attack/diabetes_experiments/number_of_targets/45/attack_data/master_challenge_train_labels.npy"

    )

    y_meta_train_5 = np.load(
        "/projects/midst-experiments/ensemble_attack/diabetes_experiments/number_of_targets/46/attack_data/master_challenge_train_labels.npy"

    )


    y_meta_train = np.concatenate([y_meta_train_1, y_meta_train_2], axis=0)

    # df_meta_test_1 = load_dataframe(
    #     Path("/projects/midst-experiments/ensemble_attack/number_of_target/42/attack_data"), "master_challenge_test.csv"
    #     )

    # df_meta_test_2 = load_dataframe(
    #     Path("/projects/midst-experiments/ensemble_attack/number_of_target/44/attack_data"), "master_challenge_test.csv"
    #     )

    # df_meta_test = pd.concat([df_meta_test_1, df_meta_test_2], ignore_index=True)

    # y_meta_test_1 = np.load(
    #     "/projects/midst-experiments/ensemble_attack/number_of_target/42/attack_data/master_challenge_test_labels.npy"
    # )

    # y_meta_test_2 = np.load(
    #     "/projects/midst-experiments/ensemble_attack/number_of_target/44/attack_data/master_challenge_test_labels.npy"
    # )   

    # y_meta_test = np.concatenate([y_meta_test_1, y_meta_test_2], axis=0)

    # # Three sets of shadow models are trained separately and their paths are provided here.

    # assert len(shadow_data_paths) == 3, (
    #     "At this point of development, the shadow_data_paths list must contain exactly three elements."
    # )

    # shadow_data_collection = []

    # for model_path in shadow_data_paths:
    #     assert model_path.exists(), (
    #         f"No file found at {model_path}. Make sure the path is correct, or run shadow model training first."
    #     )

    #     with open(model_path, "rb") as f:
    #         shadow_data_and_result = pickle.load(f)
    #         shadow_data_collection.append(shadow_data_and_result)
    #         log(INFO, f"Shadow model data loaded from {model_path}.")

    # assert Path(target_model_synthetic_path).exists(), (
    #     f"No file found at {target_model_synthetic_path}. "
    #     f"Make sure the path is correct and that you have access to target model's synthetic data."
    # )

    # # Load the target model's synthetic data
    # target_synthetic_data = pd.read_csv(target_model_synthetic_path)
    # log(
    #     INFO,
    #     f"Target model's synthetic data loaded from {target_model_synthetic_path} with size {len(target_synthetic_data)}.",
    # )

    # assert target_synthetic_data is not None, "Target model's synthetic data is missing."
    # target_synthetic_data = target_synthetic_data.copy()

    # df_reference = load_dataframe(
    #     Path(config.data_paths.population_path),
    #     "population_all_with_challenge_no_id.csv",
    # )
    # log(
    #     INFO,
    #     f"Reference population data loaded from {config.data_paths.population_path} with size {len(df_reference)}.",
    # )

    # import pdb; pdb.set_trace()

    # # Extract trans_id from both train and test dataframes
    # assert "trans_id" in df_meta_train.columns, "Meta train data must have trans_id column"
    # train_trans_ids = df_meta_train["trans_id"]

    # assert "trans_id" in df_meta_test.columns, "Meta test data must have trans_id column"
    # test_trans_ids = df_meta_test["trans_id"]

    # df_meta_train = df_meta_train.drop(columns=["trans_id", "account_id"])
    # df_meta_test = df_meta_test.drop(columns=["trans_id", "account_id"])

    # Fit the metaclassifier.
    meta_classifier_type = MetaClassifierType(config.metaclassifier.model_type)

    # 1. Initialize the attacker
    blending_attacker = BlendingPlusPlus(
        config=config,
        # shadow_data_collection=shadow_data_collection,
        data_types_file_path=Path(config.metaclassifier.data_types_file_path),
        meta_classifier_type=meta_classifier_type,
        random_seed=config.random_seed,
    )

    log(INFO, f"{meta_classifier_type} created with random seed {config.random_seed}.")

    # Extract and save the meta features for the meta train set

    # meta_features = blending_attacker.prepare_meta_features(
    #         df_input=df_meta_train,
    #         df_synthetic=target_synthetic_data,
    #         df_reference=df_reference,
    #         id_column_data=train_trans_ids,
    #         categorical_cols=blending_attacker.column_types["categorical"],
    #         numerical_cols=blending_attacker.column_types["numerical"],
    #         id_column_name=blending_attacker.column_types["id_column_name"],
    #     )
    
    # # Create a directory to save the extracted features
    # features_dir = Path(config.data_paths.processed_attack_data_path) / "extracted_features"
    # features_dir.mkdir(parents=True, exist_ok=True)
    # file_name = "meta_train_features_" + str(config.random_seed) + ".csv"
    # meta_features.to_csv(features_dir / file_name, index=False)
    # log(INFO, f"Meta features for meta train set extracted and saved at {features_dir / file_name}.")

    # load the extracted meta features for the meta train set
    features_dir_1 = "/projects/midst-experiments/ensemble_attack/diabetes_experiments/number_of_targets/extracted_features/meta_train_features_42.csv"
    df_meta_train_1 = pd.read_csv(features_dir_1)

    features_dir_2 = "/projects/midst-experiments/ensemble_attack/diabetes_experiments/number_of_targets/extracted_features/meta_train_features_43.csv"
    df_meta_train_2 = pd.read_csv(features_dir_2)

    features_dir_3 = "/projects/midst-experiments/ensemble_attack/diabetes_experiments/number_of_targets/extracted_features/meta_train_features_44.csv"
    df_meta_train_3 = pd.read_csv(features_dir_3)

    features_dir_4 = "/projects/midst-experiments/ensemble_attack/diabetes_experiments/number_of_targets/extracted_features/meta_train_features_45.csv"
    df_meta_train_4 = pd.read_csv(features_dir_4)

    features_dir_5 = "/projects/midst-experiments/ensemble_attack/diabetes_experiments/number_of_targets/extracted_features/meta_train_features_46.csv"
    df_meta_train_5 = pd.read_csv(features_dir_5)

    #concatenate the two dataframes
    df_meta_train = pd.concat([df_meta_train_1, df_meta_train_2], ignore_index=True)

    # 2. Train the attacker on the meta-train set
    blending_attacker.fit(
        df_train=df_meta_train,
        y_train=y_meta_train,
        # df_target_synthetic=target_synthetic_data,
        # df_reference=df_reference,
        # id_column_data=train_trans_ids,
        use_gpu=config.metaclassifier.use_gpu,
        epochs=config.metaclassifier.epochs,
    )

    model_filename = "42_43_metaclassifier" + str(config.random_seed)
    model_path = Path(config.model_paths.metaclassifier_model_path) / f"{model_filename}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(blending_attacker.trained_model, f)

    log(INFO, f"Metaclassifier model saved to {model_path}")

    # 3. Get predictions on the meta test set (evaluation of the trained metaclassifier)
    # For evaluation, we test the meta classifier on the meta test set provided the target's synthetic data.
    # probabilities, pred_score = blending_attacker.predict(
    #     df_test=df_meta_test,
    #     df_original_synthetic=target_synthetic_data,  # For evaluation only
    #     df_reference=df_reference,
    #     id_column_data=test_trans_ids,
    #     y_test=y_meta_test,
    # )

    # # Save the evaluation prediction probabilities
    # attack_evaluation_result_path = Path(config.data_paths.attack_evaluation_result_path)
    # attack_evaluation_result_path.mkdir(parents=True, exist_ok=True)
    # file_name = attack_evaluation_result_path / f"{model_filename}_val_pred_proba.npy"
    # np.save(file_name, probabilities)
    # log(INFO, f"Evaluation prediction probabilities saved at {file_name}.")

    # if pred_score is not None:
    #     log(INFO, f"TPR at FPR=0.1: {pred_score:.4f}")