from logging import INFO
from pathlib import Path

from omegaconf import DictConfig

from midst_toolkit.attacks.ensemble.data_utils import load_dataframe
from midst_toolkit.attacks.ensemble.rmia.shadow_model_training import (
    train_three_sets_of_shadow_models,
)
from midst_toolkit.common.logger import log


def run_shadow_model_training(config: DictConfig) -> list[Path]:
    """
    Function to run the shadow model training for RMIA attack.

    Args:
        config: Configuration object set in config.yaml.

    Returns:
        Paths to the saved shadow model results for the three sets of shadow models.
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
    assert "trans_id" in df_master_challenge_train.columns
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

    return [first_set_result_path, second_set_result_path, third_set_result_path]
