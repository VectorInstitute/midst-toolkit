"""
This file is an uncompleted example script for running the Ensemble Attack on MIDST challenge
provided resources and data.
"""

import pickle
from datetime import datetime
from logging import INFO
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from examples.ensemble_attack.real_data_collection import collect_population_data_ensemble
from midst_toolkit.attacks.ensemble.blending import BlendingPlusPlus, MetaClassifierType
from midst_toolkit.attacks.ensemble.data_utils import load_dataframe
from midst_toolkit.attacks.ensemble.process_split_data import process_split_data
from midst_toolkit.attacks.ensemble.rmia.shadow_model_training import (
    train_three_sets_of_shadow_models,
)
from midst_toolkit.common.logger import log


def run_data_processing(config: DictConfig) -> None:
    """
    Function to run the data processing pipeline.

    Args:
        config: Configuration object set in config.yaml.
    """
    log(INFO, "Running data processing pipeline...")
    # Collect the real data from the MIDST challenge resources.
    population_data = collect_population_data_ensemble(
        midst_data_input_dir=Path(config.data_paths.midst_data_path),
        data_processing_config=config.data_processing_config,
        save_dir=Path(config.data_paths.population_path),
    )
    # The following function saves the required dataframe splits in the specified processed_attack_data_path path.
    process_split_data(
        all_population_data=population_data,
        processed_attack_data_path=Path(config.data_paths.processed_attack_data_path),
        # TODO: column_to_stratify value is not documented in the original codebase.
        column_to_stratify=config.data_processing_config.column_to_stratify,
        num_total_samples=config.data_processing_config.population_sample_size,
        random_seed=config.random_seed,
    )
    log(INFO, "Data processing pipeline finished.")


def run_shadow_model_training(config: DictConfig) -> None:
    """
    Function to run the shadow model training for RMIA attack.

    Args:
        config: Configuration object set in config.yaml.
    """
    log(INFO, "Running shadow model training...")
    # Load the required dataframes for shadow model training.
    # For shadow model training we need master_challenge_train and population data.
    # Master challenge is the main training (or fine-tuning) data for the shadow models.
    df_master_challenge_train = load_dataframe(
        Path(config.data_paths.processed_attack_data_path),
        "master_challenge_train.csv",
    )
    # Population data is used to pre-train some of the shadow models.
    df_population_with_challenge = load_dataframe(
        Path(config.data_paths.population_path),
        "population_all_with_challenge.csv",
    )
    # Make sure master challenge train and population data have the "trans_id" column.
    assert "trans_id" in df_master_challenge_train.columns, (
        "trans_id column should be present in master train data for the shadow model pipeline."
    )
    assert "trans_id" in df_population_with_challenge.columns
    # ``population_data`` in ensemble attack is used for shadow pre-training, and
    # ``master_challenge_df`` is used for fine-tuning for half of the shadow models.
    # For the other half of the shadow models, only ``master_challenge_df`` is used for training.
    first_set_result_path, second_set_result_path, third_set_result_path = train_three_sets_of_shadow_models(
        population_data=df_population_with_challenge,
        master_challenge_data=df_master_challenge_train,
        shadow_models_output_path=Path(config.shadow_training.shadow_models_output_path),
        training_json_config_paths=config.shadow_training.training_json_config_paths,
        fine_tuning_config=config.shadow_training.fine_tuning_config,
        table_name="trans",
        id_column_name="trans_id",
        # Number of shadow models to train in each set of shadow training (3 sets total) results in
        # ``4 * n_models_per_set`` total shadow models.
        n_models_per_set=4,  # 4 based on the original code, must be even
        n_reps=12,  # Number of repetitions of challenge points in each shadow model training set. `12` based on the original code
        random_seed=config.random_seed,
    )
    log(
        INFO,
        f"Shadow model training finished and saved at 1) {first_set_result_path}, 2) {second_set_result_path}, 3) {third_set_result_path}",
    )


def run_metaclassifier_training(config: DictConfig) -> None:
    """
    Fuction to run the metaclassifier training and evaluation.

    Args:
        config: Configuration object set in config.yaml.
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

    # Synthetic data borrowed from the attack implementation repository.
    # From (https://github.com/CRCHUM-CITADEL/ensemble-mia/tree/main/input/tabddpm_black_box/meta_classifier)
    # TODO: Change this file path to the path where the synthetic data is stored.
    df_synthetic = load_dataframe(
        Path(config.data_paths.processed_attack_data_path),
        "synth.csv",
    )

    df_reference = load_dataframe(
        Path(config.data_paths.population_path),
        "population_all_with_challenge_no_id.csv",
    )

    # Fit the metaclassifier.
    meta_classifier_enum = MetaClassifierType(config.metaclassifier.model_type)

    # 1. Initialize the attacker
    blending_attacker = BlendingPlusPlus(
        config=config, meta_classifier_type=meta_classifier_enum, random_seed=config.random_seed
    )
    log(INFO, f"{meta_classifier_enum} created with random seed {config.random_seed}, starting training...")

    # 2. Train the attacker on the meta-train set

    blending_attacker.fit(
        df_train=df_meta_train,
        y_train=y_meta_train,
        df_synthetic=df_synthetic,
        df_reference=df_reference,
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


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """
    Run the Ensemble Attack example pipeline.
    As the first step, data processing is done.
    Second step is shadow model training used for RMIA attack.
    Third step is metaclassifier training and evaluation.

    Args:
        config: Attack configuration as an OmegaConf DictConfig object.
    """
    if config.pipeline.run_data_processing:
        run_data_processing(config)
    if config.pipeline.run_shadow_model_training:
        run_shadow_model_training(config)
    if config.pipeline.run_metaclassifier_training:
        run_metaclassifier_training(config)


if __name__ == "__main__":
    main()
