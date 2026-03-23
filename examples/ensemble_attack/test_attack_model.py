"""This script loads the trained attack model and performs the attack on a target model given its synthetic data."""

import json
import pickle
from enum import Enum
from logging import INFO
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from examples.ensemble_attack.real_data_collection import (
    COLLECTED_DATA_FILE_NAME,
    AttackDataset,
    AttackType,
    collect_midst_data,
)
from examples.ensemble_attack.run_shadow_model_training import run_shadow_model_training
from midst_toolkit.attacks.ensemble.blending import BlendingPlusPlus, MetaClassifierType
from midst_toolkit.attacks.ensemble.data_utils import load_dataframe
from midst_toolkit.attacks.ensemble.models import (
    EnsembleAttackModelRunner,
    EnsembleAttackTabDDPMModelRunner,
    EnsembleAttackTabDDPMTrainingConfig,
)
from midst_toolkit.attacks.ensemble.process_split_data import PROCESSED_TRAIN_DATA_FILE_NAME
from midst_toolkit.common.logger import log
from midst_toolkit.common.random import set_all_random_seeds
from midst_toolkit.models.clavaddpm.train import get_df_without_id


class RmiaTrainingDataChoice(Enum):
    ONLY_CHALLENGE = "only_challenge"
    ONLY_TRAIN = "only_train"
    COMBINED = "combined"


def save_results(
    attack_results_path: Path, metaclassifier_model_name: str, probabilities: np.ndarray, pred_score: float | None
) -> None:
    """
    Saves the test prediction probabilities and metric results.

    Args:
        attack_results_path: Path to save the attack results.
        metaclassifier_model_name: Name of the metaclassifier model to be used to name score and prediction files.
        probabilities: Prediction probabilities from the metaclassifier.
        pred_score: Prediction score to be saved.
    """
    file_name = attack_results_path / f"{metaclassifier_model_name}_test_pred_proba.npy"
    np.save(file_name, probabilities)
    log(INFO, f"Test prediction probabilities saved at {file_name}.")

    if pred_score is not None:
        log(INFO, f"TPR at FPR=0.1: {pred_score:.4f}")

        # Save the metric results into a text file.
        metric_save_path = attack_results_path / f"prediction_score_{metaclassifier_model_name}.txt"
        with open(metric_save_path, "w") as f:
            f.write(f"TPR at FPR=0.1: {pred_score:.4f}\n")


def extract_primary_id_column(
    data_frame: pd.DataFrame,
    data_types_file_path: Path,
) -> pd.Series:
    """
    Extracts and returns the primary IDs from the dataframe. The primary ID column is identified based on
    the data types JSON file with "id_column_name" key.
    primary IDs are unique keys in the dataset.
    For example, in the Berka dataset, "trans_id" is the primary ID column, while "account_id" is not.

    Args:
        data_frame: Input dataframe.
        data_types_file_path: Path to the data types JSON file.

    Returns:
        A Series containing the extracted data of the main ID column.
    """
    # Extract ID column from the dataframe
    with open(data_types_file_path, "r") as f:
        column_types = json.load(f)

    assert "id_column_name" in column_types, f"{data_types_file_path} must contain 'id_column_name' key."
    id_column_name = column_types["id_column_name"]
    # Make sure we have one main id column
    assert isinstance(id_column_name, str), "Only one main id column should be identified."
    assert id_column_name in data_frame.columns, f"Dataframe must have {id_column_name} column"

    return data_frame[id_column_name]


def run_rmia_shadow_training(
    model_runner: EnsembleAttackModelRunner,
    config: DictConfig,
    df_challenge: pd.DataFrame,
) -> list[dict[str, list[Any]]]:
    """
    Three sets of shadow models will be trained as a part of this attack.
    Note that shadow models need to be trained on the collection of challenge points once and used
    for all the target models in a setting. In other words, in a standard setting, the
    testing points (experiment challenge points) are used as training or included in training data
    of the shadow models, and these shadow models are used to attack all target models.

    Args:
        model_runner: The model runner to be used for training the shadow models.
            Should be an instance of `EnsembleAttackModelRunner`.
        config: Configuration object set in config.yaml.
        df_challenge: DataFrame containing the challenge data points for shadow model training.

    Return:
        A list containing three dictionaries, each representing a collection of shadow
            models with their training data and generated synthetic outputs.
    """
    shadow_model_paths = run_shadow_model_training(model_runner, config, df_challenge_train=df_challenge)

    assert len(shadow_model_paths) == 3, "For testing, meta classifier needs the path to three sets of shadow models."

    # Initialize an empty list to store the data and results of shadow models after loading.
    shadow_data_collection = []
    for model_path in shadow_model_paths:
        assert model_path.exists(), (
            f"No file found at {model_path}. Make sure the path is correct, or run shadow model training first."
        )

        with open(model_path, "rb") as f:
            shadow_data_and_result = pickle.load(f)
            shadow_data_collection.append(shadow_data_and_result)

    return shadow_data_collection


def load_trained_rmia_shadows_for_test_phase(
    shadow_data_paths: list[Path],
) -> tuple[list[dict[str, list[Any]]], bool]:
    """
    Loads previously trained RMIA shadow models for the testing phase. Makes sure
    all shadow models exist before loading. Otherwise, returns an empty list and False.
    If False is returned, all shadow models need to be trained for the testing phase.

    Args:
        shadow_data_paths: List of paths to the saved shadow model data.

    Returns:
        A tuple containing:
            - A list of dictionaries, each representing a collection of shadow
                models with their training data and generated synthetic outputs.
            - A boolean indicating whether all shadow models were successfully loaded.
    """
    shadow_data_collection = []
    # models_exists is a list indicating the existence of each RMIA shadow model
    models_exists = []
    for model_path in shadow_data_paths:
        if model_path.exists():
            with open(model_path, "rb") as f:
                shadow_data_and_result = pickle.load(f)
                shadow_data_collection.append(shadow_data_and_result)
            log(INFO, f"Loaded existing shadow model at {model_path}.")
            models_exists.append(True)
        else:
            models_exists.append(False)
            log(INFO, f"No shadow model found at {model_path}. Need to train RMIA shadow models for testing phase.")
    if not all(models_exists):
        # If even one of the models does not exist, return empty collection and False to indicate training is needed.
        shadow_data_collection = []
    return shadow_data_collection, all(models_exists)


def collect_challenge_and_train_data(
    data_processing_config: DictConfig, processed_attack_data_path: Path, targets_data_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collect experiment's challenge dataset (``df_challenge_experiment``) and master train data (``df_master_train``).
    Experiment's challenge data refers to the collection of all the challenge points of all the target models that
    we want to attack in this experiment.
    Master train data refers to the dataset that was used to train the metaclassifier.

    To collect experiment's challenge data, the ``collect_midst_data`` function is called. This function
    reads all the challenge datasets under each target model's folder.
    Refer to `real_data_collection.collect_midst_data` for more information.


    Args:
        data_processing_config: Configuration object for data processing.
        processed_attack_data_path: Path to the processed attack data.
        targets_data_path: Path to the target model's data.

    Returns:
        Tuple of (df_challenge_experiment, df_master_train).
    """
    # Collect all repo's challenge points
    challenge_attack_names = data_processing_config.challenge_attack_data_types_to_collect
    challenge_attack_types = [AttackType(attack_name) for attack_name in challenge_attack_names]
    df_challenge_experiment = collect_midst_data(
        midst_data_input_dir=targets_data_path,
        attack_types=challenge_attack_types,
        # For ensemble experiments, change to ``test`` for 10k, and change to ``final`` for 20k
        split_folders=["final"],
        dataset=AttackDataset.CHALLENGE,
        data_processing_config=data_processing_config,
    )
    log(
        INFO,
        f"Collected challenge data length: {len(df_challenge_experiment)} for the testing phase's shadow training.",
    )

    # Load master challenge train data
    df_master_train = load_dataframe(
        processed_attack_data_path,
        PROCESSED_TRAIN_DATA_FILE_NAME,
    )
    log(
        INFO,
        f"Loaded master challenge train data length: {len(df_master_train)} for the testing phase's shadow training.",
    )

    return df_challenge_experiment, df_master_train


def select_challenge_data_for_training(
    attack_rmia_shadow_training_data_choice: RmiaTrainingDataChoice,
    df_challenge_experiment: pd.DataFrame,
    df_master_train: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the appropriate challenge data based on config choice.

    Args:
        attack_rmia_shadow_training_data_choice: Strategy for creating challenge train data for RMIA shadow training.
            It can be one of the following:
            - "only_challenge": Use only challenge experiment data (``df_challenge_experiment``).
            - "only_train": Use only master train data (``df_master_train``). Note that this option contradicts
                with the original design and purpose of training RMIA shadow models on the challenge points as
                RMIA signals (IN train signals) for challenge points could only be computed if
                shadow models are trained on these points.
            - "combined": Combine both challenge experiment data and master train data. This can
                potentially be advantageous based on the experiments as RMIA shadows are trained on
                more data points.
        df_challenge_experiment: Challenge points in this experiment.
        df_master_train: Master train data used to train the meta classifier.

    Raises:
        ValueError: If an invalid choice is provided.

    Returns:
        Selected challenge data.
    """
    if attack_rmia_shadow_training_data_choice == RmiaTrainingDataChoice.COMBINED:
        # Run RMIA shadow model training on experiments challenge points + master challenge train data
        df_challenge = pd.concat([df_challenge_experiment, df_master_train]).drop_duplicates()
        log(INFO, f"Combined challenge data length for RMIA shadow training: {len(df_challenge)}.")
    elif attack_rmia_shadow_training_data_choice == RmiaTrainingDataChoice.ONLY_CHALLENGE:
        df_challenge = df_challenge_experiment
        log(INFO, "Using only challenge data points for RMIA shadow training.")
    elif attack_rmia_shadow_training_data_choice == RmiaTrainingDataChoice.ONLY_TRAIN:
        df_challenge = df_master_train
        log(INFO, "Using only master challenge train data points for RMIA shadow training.")
    else:
        raise ValueError(
            "Invalid choice for attack_rmia_shadow_training_data_choice. Must be one of 'combined', 'only_challenge', or 'only_train'."
        )

    return df_challenge


def train_rmia_shadows_for_test_phase(
    model_runner: EnsembleAttackModelRunner,
    config: DictConfig,
) -> list[dict[str, list[Any]]]:
    """
    Function to train RMIA shadow models for the testing phase using the dataset containing
    challenge data points.

    Args:
        model_runner: The model runner to be used for training the shadow models.
            Should be an instance of `EnsembleAttackModelRunner`.
        config: Configuration object set in ``experiments_config.yaml``.

    Returns:
        A list containing three dictionaries, each representing a collection of shadow
            models with their training data IDs and generated synthetic outputs.
    """
    # Checking if challenge data exists
    processed_attack_data_path = Path(config.data_paths.processed_attack_data_path)
    data_file_name = config.data_file_name if "data_file_name" in config else COLLECTED_DATA_FILE_NAME
    challenge_data_file_name = f"{Path(data_file_name).stem}_challenge_data.csv"

    if (processed_attack_data_path / challenge_data_file_name).exists():
        log(INFO, "Skipping data collection for testing phase.")
        df_challenge_experiment = load_dataframe(
            processed_attack_data_path,
            challenge_data_file_name,
        )
        df_master_train = load_dataframe(
            processed_attack_data_path,
            PROCESSED_TRAIN_DATA_FILE_NAME,
        )
    else:
        # If challenge data does not exist, collect it from the cluster
        df_challenge_experiment, df_master_train = collect_challenge_and_train_data(
            config.data_processing_config,
            processed_attack_data_path=Path(config.data_paths.processed_attack_data_path),
            targets_data_path=Path(config.data_processing_config.midst_data_path),
        )

    # Load the challenge dataframe for training RMIA shadow models.
    rmia_training_choice = RmiaTrainingDataChoice(config.target_model.attack_rmia_shadow_training_data_choice)
    df_challenge = select_challenge_data_for_training(rmia_training_choice, df_challenge_experiment, df_master_train)
    return run_rmia_shadow_training(model_runner, config, df_challenge=df_challenge)


# TODO: consider moving this and potentially the other functions above to the main library.
def run_metaclassifier_testing(model_runner: EnsembleAttackModelRunner, config: DictConfig) -> None:
    """
    Function to run the attack on a single target model using a trained metaclassifier.
    Note that RMIA shadow models need to be trained for every new set of target models on
    their collected challenge data, but once they are trained on all the experiment challenge points,
    for the first target, we can reuse them for the other targets in the same experiment.
    Unlike the training phase, in the testing phase, we don't need to train a target shadow model
    since we already have access to the synthetic data of a real target model.
    All the collected population data that is used for training, is still needed during testing to compute some
    of the signals (DOMIAS).
    Test prediction probabilities are saved to the specified attack result path in the config.

    Args:
        model_runner: The model runner to be used for testing the metaclassifier.
            Should be an instance of `EnsembleAttackModelRunner`.
        config: Configuration object set in ``experiments_config.yaml``.
    """
    log(
        INFO,
        f"Running metaclassifier testing on target synthetic data at {config.target_model.target_synthetic_data_path}...",
    )

    if config.random_seed is not None:
        set_all_random_seeds(seed=config.random_seed)
        log(INFO, f"Testing phase random seed set to {config.random_seed}.")

    # 1) Load the trained metaclassifier model to make sure it exists before proceeding.
    meta_classifier_type = MetaClassifierType(config.metaclassifier.model_type)

    metaclassifier_model_name = config.metaclassifier.meta_classifier_model_name
    mataclassifier_path = Path(config.metaclassifier.metaclassifier_model_path) / f"{metaclassifier_model_name}.pkl"
    assert mataclassifier_path.exists(), (
        f"No metaclassifier model found at {mataclassifier_path}. Make sure to run the training script first."
    )

    with open(mataclassifier_path, "rb") as f:
        trained_mataclassifier_model = pickle.load(f)

    log(INFO, f"Metaclassifier model loaded from {mataclassifier_path}, starting the test...")

    # 2) Read target model's challenge data and synthetic data.
    # Back-box attacker has only access to the target model's synthetic data and challenge points.
    # We also load challenge labels to report the attack performance at the end.
    challenge_data_path = Path(config.target_model.challenge_data_path)
    challenge_label_path = Path(config.target_model.challenge_label_path)

    test_data = pd.read_csv(challenge_data_path)
    log(INFO, f"Challenge data loaded from {challenge_data_path} with a size of {len(test_data)}.")

    if challenge_label_path.suffix == ".npy":
        test_target = np.load(challenge_label_path).squeeze()
    elif challenge_label_path.suffix == ".csv":
        test_target = pd.read_csv(challenge_label_path).to_numpy().squeeze()
    else:
        raise ValueError(f"Unsupported challenge label file type: {challenge_label_path}. Must be .npy or .csv.")

    assert len(test_data) == len(test_target), "Number of challenge labels must match number of challenge data points."

    target_synthetic_path = Path(config.target_model.target_synthetic_data_path)
    target_synthetic_data = pd.read_csv(target_synthetic_path)
    log(
        INFO, f"Target synthetic data loaded from {target_synthetic_path} with a size of {len(target_synthetic_data)}."
    )

    # If the synthetic data has more points than specified in the config, take only the required number.
    if len(target_synthetic_data) > config.shadow_training.number_of_points_to_synthesize:
        # Take only the required number of synthetic data points
        target_synthetic_data = target_synthetic_data.head(config.shadow_training.number_of_points_to_synthesize)
        log(INFO, f"Target synthetic data size adjusted to {len(target_synthetic_data)} based on the config setting.")

    # 3) Shadow Model Training Step.
    # Make sure to assign a new path for shadow models trained for target's challenge points to
    # avoid overriding train's shadow models.
    # TODO: Assign specific shadow collection path for test phase.
    config.shadow_training.shadow_models_output_path = config.target_model.target_shadow_models_output_path
    shadow_data_paths = [Path(path) for path in config.shadow_training.final_shadow_models_path]
    assert len(shadow_data_paths) == 3, "The attack_data_paths list must contain exactly three elements."

    # If shadows are already trained for test (``models_exists`` is True), don't need to train again.
    # Load shadow training collection from previously trained shadow models.
    shadow_data_collection, models_exists = load_trained_rmia_shadows_for_test_phase(shadow_data_paths)

    if not models_exists:
        log(INFO, "Shadow models for testing phase do not exist. Training RMIA shadow models...")
        shadow_data_collection = train_rmia_shadows_for_test_phase(model_runner, config)

    else:
        log(INFO, "All shadow models for testing phase found. Using existing RMIA shadow models...")

    # Extract the main ID column's values from the test data
    test_trans_ids = extract_primary_id_column(
        data_frame=test_data,
        data_types_file_path=Path(config.metaclassifier.data_types_file_path),
    )
    # Drop id columns from the test data. Berka has two id columns: "trans_id" and "account_id".
    test_data = get_df_without_id(test_data)

    # 4) Initialize the attacker object, and assign the loaded metaclassifier to it.
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

    # Load the reference population data for DOMIAS signals.
    data_file_name = config.data_file_name if "data_file_name" in config else COLLECTED_DATA_FILE_NAME
    df_reference = load_dataframe(Path(config.data_paths.population_path), f"{Path(data_file_name).stem}_no_id.csv")

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
    save_results(attack_results_path, metaclassifier_model_name, probabilities, pred_score)


# TODO: Perform inference on all the target models sequentially in a single run instead of running this script
# multiple times. For more information, refer to https://app.clickup.com/t/868h4xk86
@hydra.main(config_path="configs", config_name="experiment_config", version_base=None)
def run_metaclassifier_testing_with_tabddpm(config: DictConfig) -> None:
    """
    Run the attack on a single target model using a trained metaclassifier.
    RMIA shadow models will be trained using the TabDDPM model.

    Args:
        config: Configuration object set in config.yaml.
    """
    log(INFO, "Running metaclassifier testing with TabDDPM...")

    with open(config.shadow_training.training_json_config_paths.training_config_path, "r") as file:
        training_config = EnsembleAttackTabDDPMTrainingConfig(**json.load(file))
    fine_tune_diffusion_iterations = config.shadow_training.fine_tuning_config.fine_tune_diffusion_iterations
    training_config.fine_tuning_diffusion_iterations = fine_tune_diffusion_iterations
    fine_tune_classifier_iterations = config.shadow_training.fine_tuning_config.fine_tune_classifier_iterations
    training_config.fine_tuning_classifier_iterations = fine_tune_classifier_iterations
    training_config.number_of_points_to_synthesize = config.shadow_training.number_of_points_to_synthesize

    model_runner = EnsembleAttackTabDDPMModelRunner(training_config=training_config)

    run_metaclassifier_testing(model_runner, config)


if __name__ == "__main__":
    run_metaclassifier_testing_with_tabddpm()
