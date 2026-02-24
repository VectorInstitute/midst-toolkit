import shutil
from logging import INFO
from pathlib import Path
from typing import cast

from omegaconf import DictConfig

from midst_toolkit.attacks.ensemble.data_utils import load_dataframe
from midst_toolkit.attacks.ensemble.rmia.shadow_model_training import (
    train_three_sets_of_shadow_models,
)
from midst_toolkit.attacks.ensemble.shadow_model_utils import (
    ModelType,
    TrainingResult,
    save_additional_training_config,
    train_or_fine_tune_ctgan,
    train_tabddpm_and_synthesize,
)
from midst_toolkit.common.config import ClavaDDPMTrainingConfig, CTGANTrainingConfig
from midst_toolkit.common.logger import log


DEFAULT_TABLE_NAME = "trans"
DEFAULT_ID_COLUMN_NAME = "trans_id"
DEFAULT_MODEL_TYPE = ModelType.TABDDPM


def run_target_model_training(config: DictConfig) -> Path:
    """
    Function to run the target model training for RMIA attack.

    Args:
        config: Configuration object set in config.yaml.

    Returns:
        Path to the saved target model's synthetic data.
    """
    log(INFO, "Running target model training...")

    # Load the required dataframe for target model training.
    df_real_data = load_dataframe(
        Path(config.data_paths.processed_attack_data_path),
        "real_train.csv",
    )

    # TODO: Test when pipeline is complete to make sure real_data is correct.

    target_model_output_path = Path(config.shadow_training.target_model_output_path)
    target_training_json_config_paths = config.shadow_training.training_json_config_paths

    table_name = config.table_name if "table_name" in config else DEFAULT_TABLE_NAME

    target_folder = target_model_output_path / "target_model"

    model_type = DEFAULT_MODEL_TYPE
    if "model_name" in config.shadow_training:
        model_type = ModelType(config.shadow_training.model_name)
    log(INFO, f"Training target model with model type: {model_type.value}")

    target_folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        target_training_json_config_paths.table_domain_file_path,
        target_folder / f"{table_name}_domain.json",
    )
    shutil.copyfile(
        target_training_json_config_paths.dataset_meta_file_path,
        target_folder / "dataset_meta.json",
    )
    configs, save_dir = save_additional_training_config(
        data_dir=target_folder,
        training_config_json_path=Path(target_training_json_config_paths.training_config_path),
        final_config_json_path=target_folder / f"{table_name}.json",  # Path to the new json
        experiment_name="trained_target_model",
        model_type=model_type,
    )

    train_result: TrainingResult
    if model_type == ModelType.TABDDPM:
        train_result = train_tabddpm_and_synthesize(
            train_set=df_real_data,
            configs=cast(ClavaDDPMTrainingConfig, configs),
            save_dir=save_dir,
            synthesize=True,
            number_of_points_to_synthesize=config.shadow_training.number_of_points_to_synthesize,
        )
    elif model_type == ModelType.CTGAN:
        train_result = train_or_fine_tune_ctgan(
            dataset=df_real_data,
            configs=cast(CTGANTrainingConfig, configs),
            save_dir=save_dir,
            synthesize=True,
        )

    # To train the attack model (metaclassifier), we only need to save target's synthetic data,
    # and not the entire target model's training result object.
    assert train_result.synthetic_data is not None, "Target model synthetic data is not generated successfully."
    target_synthetic_data = train_result.synthetic_data

    # Save the target model's synthetic data
    target_model_synthetic_path = config.shadow_training.target_synthetic_data_path
    target_synthetic_data.to_csv(target_model_synthetic_path, index=False)

    return target_model_synthetic_path


def run_shadow_model_training(config: DictConfig) -> list[Path]:
    """
    Function to run the shadow model training for RMIA attack.

    Args:
        config: Configuration object set in config.yaml.

    Returns:
        Paths to the saved shadow model results for the three sets of shadow models. For more details,
        see the documentation and return value of `train_three_sets_of_shadow_models`
        at src/midst_toolkit/attacks/ensemble/rmia/shadow_model_training.py.
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

    table_name = config.table_name if "table_name" in config else DEFAULT_TABLE_NAME
    id_column_name = config.table_id_column_name if "table_id_column_name" in config else DEFAULT_ID_COLUMN_NAME

    model_type = DEFAULT_MODEL_TYPE
    if "model_name" in config.shadow_training:
        model_type = ModelType(config.shadow_training.model_name)
    log(INFO, f"Training shadow models with model type: {model_type.value}")

    # Make sure master challenge train and population data have the "trans_id" column.
    assert id_column_name in df_master_challenge_train.columns, (
        f"{id_column_name} column should be present in master train data for the shadow model pipeline."
    )
    assert id_column_name in df_population_with_challenge.columns, (
        f"{id_column_name} column should be present in population data for the shadow model pipeline."
    )
    assert id_column_name in df_master_challenge_train.columns, (
        f"{id_column_name} column should be present in master train data for the shadow model pipeline."
    )
    # ``population_data`` in ensemble attack is used for shadow pre-training, and
    # ``master_challenge_df`` is used for fine-tuning for half of the shadow models.
    # For the other half of the shadow models, only ``master_challenge_df`` is used for training.
    first_set_result_path, second_set_result_path, third_set_result_path = train_three_sets_of_shadow_models(
        population_data=df_population_with_challenge,
        master_challenge_data=df_master_challenge_train,
        shadow_models_output_path=Path(config.shadow_training.shadow_models_output_path),
        training_json_config_paths=config.shadow_training.training_json_config_paths,
        fine_tuning_config=config.shadow_training.fine_tuning_config,
        table_name=table_name,
        id_column_name=id_column_name,
        # Number of shadow models to train in each set of shadow training (3 sets total) results in
        # ``4 * n_models_per_set`` total shadow models.
        n_models_per_set=4,  # 4 based on the original code, must be even
        n_reps=12,  # Number of repetitions of challenge points in each shadow model training set. `12` based on the original code
        number_of_points_to_synthesize=config.shadow_training.number_of_points_to_synthesize,
        random_seed=config.random_seed,
        model_type=model_type,
    )
    log(
        INFO,
        f"Shadow model training finished and saved at \n1) {first_set_result_path} \n2) {second_set_result_path} \n3) {third_set_result_path}",
    )

    return [first_set_result_path, second_set_result_path, third_set_result_path]
