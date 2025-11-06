"""
This file is an uncompleted example script for running the EPT-MIA Attack on MIDST challenge
provided resources and data.
Overall workflow and decisions are taken with from the Cyber@BGU team's attack implementation at
https://github.com/eyalgerman/MIA-EPT.

"""

import json
from logging import INFO
from pathlib import Path

import hydra
from omegaconf import DictConfig

from midst_toolkit.attacks.ensemble.data_utils import load_dataframe, save_dataframe
from midst_toolkit.attacks.ept import feature_extraction
from midst_toolkit.common.logger import log


# Step 2 and 3: Attribute prediction model training and feature extraction
def run_attribute_prediction(config: DictConfig) -> None:
    """
    Train attribute prediction models and extract features for EPT-MIA attack.
    The function is specifically designed to work with the MIDST challenge data structure,
    and the shadow models provided by the competition organizers.
    All the reading and writing of data is handled within this function.

    Args:
        config: Configuration object set in config.yaml.
    """
    log(INFO, "Running attribute prediction model training.")

    diffusion_model_names = ["tabddpm", "tabsyn"] if config.attack_settings.single_table else ["clavaddpm"]
    modes = ["train", "dev", "final"]
    input_data_path = Path(config.data_paths.input_data_path)
    output_features_path = Path(config.data_paths.output_data_path, "attribute_prediction_features")

    # Load column types specific to the competition dataset
    with open(config.data_paths.data_types_file_path, "r") as f:
        column_types = json.load(f)

    # Iterating over directories specific to the shadow models folder structure in the competition
    for model_name in diffusion_model_names:
        model_path = Path(input_data_path / f"{model_name}_black_box")
        for mode in modes:
            current_path = Path(model_path / mode)
            model_folders = [entry.name for entry in current_path.iterdir() if entry.is_dir()]
            for model_folder in model_folders:
                # Load the data files as dataframes
                input_data_path = Path(current_path / model_folder)

                df_synthetic_data = load_dataframe(input_data_path, "trans_synthetic.csv")
                df_challenge_data = load_dataframe(input_data_path, "challenge_with_id.csv")
                # df_challenge_labels = load_dataframe(input_data_path, "challenge_label.csv")

                # Drop columns in df_syntehtic_data that end with '_id', as they do not create meaningful features
                df_synthetic_data = df_synthetic_data.drop(
                    columns=[col for col in df_synthetic_data.columns if col.endswith("_id")]
                )
                df_challenge_data = df_challenge_data.drop(
                    columns=[col for col in df_challenge_data.columns if col.endswith("_id")]
                )

                # Run feature extraction
                df_extracted_features = feature_extraction.main(
                    synthetic_data=df_synthetic_data,
                    challenge_data=df_challenge_data,
                    column_types=column_types,
                    random_seed=config.random_seed,
                )

                final_output_dir = Path(output_features_path / f"{model_name}_black_box")

                final_output_dir.mkdir(parents=True, exist_ok=True)

                # Extract the number at the end of model_folder
                model_folder_number = int(model_folder.split("_")[-1])
                file_name = f"attribute_prediction_features_{model_folder_number}.csv"

                save_dataframe(df=df_extracted_features, file_path=final_output_dir, file_name=file_name)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """
    Run the EPT-MIA Attack example pipeline.
    First step has yet to be implemented: shadow model training.
    Second and third steps are attribute prediction model training and feature extraction.

    Args:
        config: Attack configuration as an OmegaConf DictConfig object.
    """
    log(INFO, "Running EPT-MIA Attack Example Pipeline.")

    if config.attack_settings.single_table:
        log(INFO, "Data: Single-table.")
    else:
        log(INFO, "Data: Multi-table.")

    # TODO: Implement shadow model training step.

    if config.pipeline.run_attribute_prediction_model_training:
        run_attribute_prediction(config)


if __name__ == "__main__":
    main()
