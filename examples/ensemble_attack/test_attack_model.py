"""This script loads the trained attack model and performs the attack on a target model given its synthetic data."""

import json
import pickle
from logging import INFO
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from examples.ensemble_attack.run_shadow_model_training import run_shadow_model_training
from examples.ensemble_attack.real_data_collection import collect_midst_data, AttackType
from midst_toolkit.attacks.ensemble.blending import BlendingPlusPlus, MetaClassifierType
from midst_toolkit.attacks.ensemble.data_utils import load_dataframe
from midst_toolkit.common.logger import log
from midst_toolkit.common.random import set_all_random_seeds



def run_rmia_shadow_training(config: DictConfig, df_challenge) -> list[dict[str, list[Any]]]:
    """
    Three sets of shadow models will be trained as a part of this attack.
    Note that for every new target model, shadow models need to be trained.
    RMIA signals (for the challenge points) are calculated based on these shadow models,
    and will be fed into the metaclassifier.

    Args:
        config: Configuration object set in ``experiments_config.yaml``.

    Return:
        A list containing three dictionaries, each representing a collection of shadow
            models with their training data and generated synthetic outputs.
    """
    shadow_model_paths = run_shadow_model_training(config, df_master_challenge_train=df_challenge)
    shadow_model_paths = [Path(path) for path in config.shadow_training.final_shadow_models_path]

    assert len(shadow_model_paths) == 3, "For testing, meta classifier needs the path to three sets of shadow models."

    shadow_data_collection = []
    for model_path in shadow_model_paths:
        assert model_path.exists(), (
            f"No file found at {model_path}. Make sure the path is correct, or run shadow model training first."
        )

        with open(model_path, "rb") as f:
            shadow_data_and_result = pickle.load(f)
            shadow_data_collection.append(shadow_data_and_result)

    return shadow_data_collection


@hydra.main(config_path="configs", config_name="experiment_config", version_base=None)
def run_metaclassifier_testing(
    config: DictConfig,
) -> None:
    """
    Function to run the attack on a target model using a trained metaclassifier.
    Note that RMIA shadow models need to be trained for every new target model's challenge dataset.
    However, we load the previously trained metaclassifier model and use it for new target models.
    Unlike the training phase, in the testing phase, we don't need to train a shadow target model
    since we already have access to the synthetic data of a real target model.
    All the collected population data that is used for training, is still needed during testing to compute some
    of the signals.
    Test prediction probabilities are saved to the specified attack result path in the config.

    Args:
        config: Configuration object set in ``experiments_config.yaml``.
    """
    log(INFO, f"Running metaclassifier testing on target model {config.target_model.target_model_id}...")

    if config.random_seed is not None:
        set_all_random_seeds(seed=config.random_seed)
        log(INFO, f"Testing phase random seed set to {config.random_seed}.")

    # 1) Load the trained metaclassifier model to make sure it exists before proceeding.
    meta_classifier_type = MetaClassifierType(config.metaclassifier.model_type)

    metaclassifier_model_name = config.metaclassifier.meta_classifier_model_name
    mataclassifier_path = Path(config.model_paths.metaclassifier_model_path) / f"{metaclassifier_model_name}.pkl"
    assert mataclassifier_path.exists(), (
        f"No metaclassifier model found at {mataclassifier_path}. Make sure to run the training script first."
    )

    with open(mataclassifier_path, "rb") as f:
        trained_mataclassifier_model = pickle.load(f)

    log(INFO, "Metaclassifier model loaded, starting the test...")

    # 2) Read target model's challenge data and synthetic data.

    # Back-box attacker has only access to the target model's synthetic data and challenge points.
    # We also load challenge labels to report the attack performance.
    challenge_data_path = Path(config.target_model.challenge_data_path)
    challenge_label_path = Path(config.target_model.challenge_label_path)

    test_data = pd.read_csv(challenge_data_path)
    log(INFO, f"Challenge data loaded from {challenge_data_path} with a size of {len(test_data)}.")

    test_target = pd.read_csv(challenge_label_path).to_numpy().squeeze()
    assert len(test_data) == len(test_target), "Number of challenge labels must match number of challenge data points."

    target_synthetic_path = Path(config.target_model.target_synthetic_data_path)
    target_synthetic_data = pd.read_csv(target_synthetic_path)
    log(
        INFO, f"Target synthetic data loaded from {target_synthetic_path} with a size of {len(target_synthetic_data)}."
    )

# 3) Shadow Model Training Step.

    # Make sure to assign a new path for shadow models trained for target's challenge points to
    # avoid overriding train's shadow models.
    config.shadow_training.shadow_models_output_path = config.target_model.target_shadow_models_output_path
    shadow_data_paths = [Path(path) for path in config.shadow_training.final_shadow_models_path]
    # if already trained for test, don't need to train again
    # Load shadow training collection from previously trained shadow models.
    assert (
        len(shadow_data_paths) == 3
    ), "The attack_data_paths list must contain exactly three elements."

    shadow_data_collection = []
    models_exist = True
    for model_path in shadow_data_paths:
        
        if model_path.exists():
            with open(model_path, "rb") as f:
                shadow_data_and_result = pickle.load(f)
                shadow_data_collection.append(shadow_data_and_result)
        else:
            models_exist = False
            break
    
    if not models_exist:
        # collect all repo's challenge points
        data_processing_config=config.data_processing_config
        challenge_attack_names = data_processing_config.challenge_attack_data_types_to_collect
        challenge_attack_types = [AttackType(attack_name) for attack_name in challenge_attack_names]
        df_challenge = collect_midst_data(
            midst_data_input_dir=Path(config.data_paths.midst_data_path),
            attack_types=challenge_attack_types,
            data_splits=["final"],  #change to test for 10k, and change to final for 20k
            dataset="challenge",
            data_processing_config=config.data_processing_config,
        )
        log(INFO, f"Collected challenge data length: {len(df_challenge)} for the testing phase's shadow training.")
        shadow_data_collection = run_rmia_shadow_training(config, df_challenge=df_challenge)




    # Extract trans_id from the test dataframe
    with open(Path(config.metaclassifier.data_types_file_path), "r") as f:
        column_types = json.load(f)
    id_column_name = column_types["id_column_name"]

    assert id_column_name in test_data.columns, "Test data must have trans_id column"
    test_trans_ids = test_data[id_column_name]

    # Drop id columns from test data
    id_column_names = [column_name for column_name in test_data.columns if column_name.endswith("_id")]
    test_data = test_data.drop(columns=id_column_names)


    


    # Load already trained shadows (only if completely are run)
    # shadow_model_paths = [Path(path) for path in config.shadow_training.final_shadow_models_path]
    # shadow_data_collection = []
    # for model_path in shadow_model_paths:
    #     assert model_path.exists(), (
    #         f"No file found at {model_path}. Make sure the path is correct, or run shadow model training first."
    #     )

    #     with open(model_path, "rb") as f:
    #         shadow_data_and_result = pickle.load(f)
    #         shadow_data_collection.append(shadow_data_and_result)



    # 4) Initialize the attacker object, and assign the loaded metaclassifier to it.

    df_reference = load_dataframe(
        Path(config.data_paths.population_path),
        "population_all_with_challenge_no_id.csv",
    )

    blending_attacker = BlendingPlusPlus(
        config=config,
        shadow_data_collection=shadow_data_collection,
        data_types_file_path=Path(config.metaclassifier.data_types_file_path),
        meta_classifier_type=meta_classifier_type,
        random_seed=config.random_seed,
    )

    # Assign the trained metaclassifier model to the attacker object.
    blending_attacker.trained_model = trained_mataclassifier_model

    # 5) Get predictions on the challenge data (test set).
    probabilities, pred_score = blending_attacker.predict(
        df_test=test_data,
        df_original_synthetic=target_synthetic_data,
        df_reference=df_reference,
        id_column_data=test_trans_ids,
        y_test=test_target,
    )

    # Save the validation prediction probabilities
    attack_results_path = Path(config.target_model.attack_probabilities_result_path)
    attack_results_path.mkdir(parents=True, exist_ok=True)
    file_name = attack_results_path / f"{metaclassifier_model_name}_test_pred_proba.npy"
    np.save(file_name, probabilities)
    log(INFO, f"Test prediction probabilities saved at {file_name}.")

    if pred_score is not None:
        log(INFO, f"TPR at FPR=0.1: {pred_score:.4f}")

        # Save the metric results into a text file.
        metric_save_path = attack_results_path / f"prediction_score_{metaclassifier_model_name}.txt"
        with open(metric_save_path, "w") as f:
            f.write(f"TPR at FPR=0.1: {pred_score:.4f}\n")


if __name__ == "__main__":
    run_metaclassifier_testing()
