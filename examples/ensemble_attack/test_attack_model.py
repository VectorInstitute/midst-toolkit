# This script loads the trained attack model and performs the attack on a set of target models.

import pickle
from logging import INFO
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from examples.ensemble_attack.run_shadow_model_training import run_shadow_model_training
from midst_toolkit.attacks.ensemble.blending import BlendingPlusPlus, MetaClassifierType
from midst_toolkit.attacks.ensemble.data_utils import load_dataframe
from midst_toolkit.common.logger import log


@hydra.main(config_path="configs", config_name="experiment_config", version_base=None)
def run_metaclassifier_testing(
    config: DictConfig,
) -> None:
    """
    Function to run the attack on a target model using a trained metaclassifier.
    Note that shadow models need to be trained for every new target model's challenge dataset.
    However, we load the previously trained metaclassifier model and use it for new target models.
    Unlike the training phase, in the testing phase, we don't need to train a shadow target model
    since we already have access to the synthetic data of a real target model.
    All the collected population data that is used for training, is still needed during testing to compute some
    of the signals.


    Args:
        config: Configuration object set in experiments_config.yaml.
    """
    log(INFO, f"Running metaclassifier testing on target model {config.target_model.target_model_id}...")

    # 1) Load the trained metaclassifier model to make sure it exists before proceeding.
    meta_classifier_enum = MetaClassifierType(config.metaclassifier.model_type)

    model_name = config.metaclassifier.meta_classifier_model_name
    mataclassifier_path = Path(config.model_paths.metaclassifier_model_path) / model_name
    assert mataclassifier_path.exists(), (
        f"No metaclassifier model found at {mataclassifier_path}.\
        Make sure to run the training script first."
    )

    with open(mataclassifier_path, "rb") as f:
        trained_mataclassifier_model = pickle.load(f)

    log(INFO, "Metaclassifier model loaded, starting evaluation...")

    # 2) Read target model's challenge data and synthetic data.

    # Back-box attacker has only access to the target model's synthetic data and challenge points.
    # We also load challenge labels to report the attack performance.
    challenge_data_path = Path(config.target_model.challenge_data_path)
    challenge_label_path = Path(config.target_model.challenge_label_path)
    df_test = pd.read_csv(challenge_data_path)
    y_test = pd.read_csv(challenge_label_path).to_numpy().squeeze()

    target_synthetic_path = Path(config.target_model.target_synthetic_data_path)
    target_synthetic = pd.read_csv(target_synthetic_path)

    # Extract trans_id from the test dataframe
    assert "trans_id" in df_test.columns, "Test data must have trans_id column"
    test_trans_ids = df_test["trans_id"]
    df_test = df_test.drop(columns=["trans_id", "account_id"])

    # 3) Shadow Model Training Step.

    # Three sets of shadow models will be trained as a part of this attack.
    # Note that for every new target model, shadow models need to be trained.
    # RMIA signals (for the challenge points) are calculated based on these shadow models,
    # and will be fed into the metaclassifier.
    # Make sure to assign a new path for shadow models trained for target's challenge points to
    # avoid overriding train's shadow models.
    config.shadow_training.shadow_models_output_path = config.target_model.target_shadow_models_output_path
    shadow_model_paths = run_shadow_model_training(config)

    assert len(shadow_model_paths) == 3, "For testing, meta classifier needs the path to three sets of shadow models."

    shadow_data_collection = []
    for model_path in shadow_model_paths:
        assert model_path.exists(), (
            f"No file found at {model_path}. Make sure the path is correct, or run shadow model training first."
        )

        with open(model_path, "rb") as f:
            shadow_data_and_result = pickle.load(f)
            shadow_data_collection.append(shadow_data_and_result)

    # 4) Initialize the attacker object, and assign the loaded metaclassifier to it.
    target_synthetic = target_synthetic.copy()

    df_reference = load_dataframe(
        Path(config.data_paths.population_path),
        "population_all_with_challenge_no_id.csv",
    )

    blending_attacker = BlendingPlusPlus(
        config=config,
        shadow_data_collection=shadow_data_collection,
        data_types_file_path=Path(config.metaclassifier.data_types_file_path),
        meta_classifier_type=meta_classifier_enum,
        random_seed=config.random_seed,
    )

    # Assign the trained metaclassifier model to the attacker object.
    blending_attacker.trained_model = trained_mataclassifier_model

    # 5) Get predictions on the challenge data (test set).
    probabilities, pred_score = blending_attacker.predict(
        df_test=df_test,
        df_original_synthetic=target_synthetic,
        df_reference=df_reference,
        id_column_data=test_trans_ids,
        y_test=y_test,
    )

    # Save the validation prediction probabilities
    attack_results_path = Path(config.target_model.attack_probabilities_result_path)
    attack_results_path.mkdir(parents=True, exist_ok=True)
    np.save(
        attack_results_path / f"{config.metaclassifier.model_type}_val_pred_proba.npy",
        probabilities,
    )
    log(INFO, "Test prediction probabilities saved.")

    if pred_score is not None:
        log(INFO, f"TPR at FPR=0.1: {pred_score:.4f}")


if __name__ == "__main__":
    run_metaclassifier_testing()
