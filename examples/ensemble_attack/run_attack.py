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

    df_synth = load_dataframe(
        Path(config.data_paths.processed_attack_data_path),
        "synth.csv",
    )

    df_ref = load_dataframe(
        Path(config.data_paths.population_path),
        "population_all_with_challenge_no_id.csv",
    )

    # Fit the metaclassifier.
    meta_classifier_enum = MetaClassifierType(config.metaclassifier.model_type)

    # 1. Initialize the attacker
    blending_attacker = BlendingPlusPlus(data_configs=config.data_configs, meta_classifier_type=meta_classifier_enum)
    log(INFO, "Metaclassifier created, starting training...")

    # 2. Train the attacker on the meta-train set

    blending_attacker.fit(
        df_train=df_meta_train,
        y_train=y_meta_train,
        df_synth=df_synth,
        df_ref=df_ref,
        use_gpu=config.metaclassifier.use_gpu,
        epochs=config.metaclassifier.epochs,
    )

    log(INFO, "Metaclassifier training finished.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{timestamp}_{config.metaclassifier.model_type}_trained_metaclassifier.pkl"
    with open(Path(config.model_paths.metaclassifier_model_path) / model_filename, "wb") as f:
        pickle.dump(blending_attacker.meta_classifier_, f)

    log(INFO, "Metaclassifier model saved, starting evaluation...")

    # 3. Get predictions on the test set
    probabilities, pred_score = blending_attacker.predict(
        df_test=df_meta_test,
        df_synth=df_synth,
        df_ref=df_ref,
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

    Args:
        config: Attack configuration as an OmegaConf DictConfig object.
    """
    if config.pipeline.run_data_processing:
        run_data_processing(config)
    elif config.pipeline.run_metaclassifier_training:
        run_metaclassifier_training(config)


if __name__ == "__main__":
    main()
