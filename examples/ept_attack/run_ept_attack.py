"""
This file is an incomplete example script for running the EPT-MIA Attack on MIDST challenge
provided resources and data.
Overall workflow and decisions are taken with from the Cyber@BGU team's attack implementation at
https://github.com/eyalgerman/MIA-EPT.

"""

import json
from logging import INFO
from pathlib import Path

import hydra
from omegaconf import DictConfig

from examples.common.utils import iterate_model_folders
from midst_toolkit.attacks.ensemble.data_utils import load_dataframe, save_dataframe
from midst_toolkit.attacks.ept.feature_extraction import extract_features
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
    input_data_path = Path(config.data_paths.input_data_path)
    output_features_path = Path(config.data_paths.output_data_path, "attribute_prediction_features")

    # Load column types specific to the competition dataset
    with open(config.data_paths.data_types_file_path, "r") as f:
        column_types = json.load(f)

    # Drop columns that end with '_id' from column_types, as they do not create meaningful features
    feature_column_types = {
        "numerical": [col for col in column_types.get("numerical", []) if not col.endswith("_id")],
        "categorical": [col for col in column_types.get("categorical", []) if not col.endswith("_id")],
    }

    # Iterating over directories specific to the shadow models folder structure in the competition
    for model_name, model_data_path, model_folder in iterate_model_folders(input_data_path, diffusion_model_names):
        # Load the data files as dataframes
        df_synthetic_data = load_dataframe(model_data_path, "trans_synthetic.csv")
        df_challenge_data = load_dataframe(model_data_path, "challenge_with_id.csv")

        # Keep only the columns that are present in feature_column_types
        columns_to_keep = feature_column_types["numerical"] + feature_column_types["categorical"]
        df_synthetic_data = df_synthetic_data[columns_to_keep]
        df_challenge_data = df_challenge_data[columns_to_keep]

        # Run feature extraction
        df_extracted_features = extract_features(
            synthetic_data=df_synthetic_data,
            challenge_data=df_challenge_data,
            column_types=feature_column_types,
            random_seed=config.random_seed,
        )

        final_output_dir = output_features_path / f"{model_name}_black_box"

        final_output_dir.mkdir(parents=True, exist_ok=True)

        # Extract the number at the end of model_folder
        model_folder_number = int(model_folder.split("_")[-1])
        file_name = f"attribute_prediction_features_{model_folder_number}.csv"

        save_dataframe(df=df_extracted_features, file_path=final_output_dir, file_name=file_name)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """
    Main orchestrator of the EPT-MIA Attack example pipeline.
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

    # TODO: Implement potential data preprocessing step.
    # TODO: Implement shadow model training step.

    if config.pipeline.run_feature_extraction:
        run_attribute_prediction(config)

    # TODO: Implement attack classifier training step.


if __name__ == "__main__":
    main()
