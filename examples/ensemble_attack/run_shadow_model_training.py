import shutil
from logging import INFO
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from examples.ensemble_attack.real_data_collection import COLLECTED_DATA_FILE_NAME
from midst_toolkit.attacks.ensemble.data_utils import load_dataframe
from midst_toolkit.attacks.ensemble.model import EnsembleAttackModelRunner
from midst_toolkit.attacks.ensemble.rmia.shadow_model_training import train_three_sets_of_shadow_models
from midst_toolkit.common.logger import log


DEFAULT_TABLE_NAME = "trans"
DEFAULT_ID_COLUMN_NAME = "trans_id"


def run_target_model_training(model_runner: EnsembleAttackModelRunner, config: DictConfig) -> Path:
    """
    Function to run the target model training for RMIA attack.

    Args:
        model_runner: The model runner to be used for training the target model.
            Should be an instance of a subclass of `EnsembleAttackModelRunner`.
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

    target_folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        target_training_json_config_paths.table_domain_file_path,
        target_folder / f"{table_name}_domain.json",
    )
    shutil.copyfile(
        target_training_json_config_paths.dataset_meta_file_path,
        target_folder / "dataset_meta.json",
    )

    train_result = model_runner.train_or_fine_tune_and_synthesize(dataset=df_real_data, synthesize=True)

    # To train the attack model (metaclassifier), we only need to save target's synthetic data,
    # and not the entire target model's training result object.
    assert train_result.synthetic_data is not None, "Target model synthetic data is not generated successfully."
    target_synthetic_data = train_result.synthetic_data

    # Save the target model's synthetic data
    target_model_synthetic_path = config.shadow_training.target_synthetic_data_path
    target_synthetic_data.to_csv(target_model_synthetic_path, index=False)

    return target_model_synthetic_path


def run_shadow_model_training(
    model_runner: EnsembleAttackModelRunner, config: DictConfig, df_challenge_train: pd.DataFrame
) -> list[Path]:
    """
    Function to run the shadow model training for RMIA attack.

    Args:
        model_runner: The model runner to be used for training the shadow models. Should be an instance of
            a subclass of `EnsembleAttackModelRunner`.
        config: Configuration object set in config.yaml.
        df_challenge_train: DataFrame containing the data that is used to train RMIA shadow models.

    Returns:
        Paths to the saved shadow model results for the three sets of shadow models. For more details,
        see the documentation and return value of `train_three_sets_of_shadow_models`
        at src/midst_toolkit/attacks/ensemble/rmia/shadow_model_training.py.
    """
    log(INFO, "Running shadow model training...")

    table_name = config.table_name if "table_name" in config else DEFAULT_TABLE_NAME
    id_column_name = config.table_id_column_name if "table_id_column_name" in config else DEFAULT_ID_COLUMN_NAME
    data_file_name = config.data_file_name if "data_file_name" in config else COLLECTED_DATA_FILE_NAME

    # Load the required dataframes for shadow model training.
    # For shadow model training we need master_challenge_train and population data.
    # Master challenge is the main training (or fine-tuning) data for the shadow models.
    # Population data is used to pre-train some of the shadow models.
    df_population_with_challenge = load_dataframe(Path(config.data_paths.population_path), data_file_name)

    log(INFO, f"Training shadow models with model runner: {model_runner}")

    # Make sure master challenge train and population data have the id column.
    assert id_column_name in df_challenge_train.columns, (
        f"{id_column_name} column should be present in master train data for the shadow model pipeline."
    )
    assert id_column_name in df_population_with_challenge.columns, (
        f"{id_column_name} column should be present in population data for the shadow model pipeline."
    )
    # ``population_data`` in ensemble attack is used for shadow pre-training, and
    # ``master_challenge_df`` is used for fine-tuning for half of the shadow models.
    # For the other half of the shadow models, only ``master_challenge_df`` is used for training.
    first_set_result_path, second_set_result_path, third_set_result_path = train_three_sets_of_shadow_models(
        model_runner=model_runner,
        population_data=df_population_with_challenge,
        master_challenge_data=df_challenge_train,
        shadow_models_output_path=Path(config.shadow_training.shadow_models_output_path),
        training_json_config_paths=config.shadow_training.training_json_config_paths,
        fine_tuning_config=config.shadow_training.fine_tuning_config,
        table_name=table_name,
        id_column_name=id_column_name,
        # Number of shadow models to train in each set of shadow training (3 sets total) results in
        # ``4 * n_models_per_set`` total shadow models.
        n_models_per_set=4,  # 4 based on the original code, must be even
        n_reps=12,  # Number of repetitions of challenge points in each shadow model training set. `12` based on the original code
        random_seed=config.random_seed,
    )
    log(
        INFO,
        f"Shadow model training finished and saved at \n1) {first_set_result_path} \n2) {second_set_result_path} \n3) {third_set_result_path}",
    )

    return [first_set_result_path, second_set_result_path, third_set_result_path]
