"""
This file is an incomplete example script for running the EPT-MIA Attack on MIDST challenge
provided resources and data.
Overall workflow and decisions are taken with from the Cyber@BGU team's attack implementation at
https://github.com/eyalgerman/MIA-EPT.

"""

import itertools
import json
from datetime import datetime
from logging import INFO
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig

from examples.common.utils import iterate_model_folders
from midst_toolkit.attacks.ensemble.data_utils import load_dataframe, save_dataframe
from midst_toolkit.attacks.ept.classification import train_attack_classifier
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
    output_features_path = Path(config.data_paths.attribute_features_path)

    # Load column types specific to the competition dataset
    with open(config.data_paths.data_types_file_path, "r") as f:
        column_types = json.load(f)

    # Drop columns that end with '_id' from column_types, as they do not create meaningful features
    feature_column_types = {
        "numerical": [col for col in column_types.get("numerical", []) if not col.endswith("_id")],
        "categorical": [col for col in column_types.get("categorical", []) if not col.endswith("_id")],
    }

    # Assert that the input data path exists and is not empty
    assert input_data_path.exists() and input_data_path.is_dir(), f"Input data directory not found: {input_data_path}"
    assert any(input_data_path.iterdir()), f"Input data directory is empty: {input_data_path}"

    # Iterating over directories specific to the shadow models folder structure in the competition
    for model_name, model_data_path, model_folder, mode in iterate_model_folders(
        input_data_path, diffusion_model_names
    ):
        log(INFO, f"Processing model: {model_name}, path: {model_data_path}, folder: {model_folder}, mode: {mode}")

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

        final_output_dir = output_features_path / f"{model_name}_black_box" / mode

        final_output_dir.mkdir(parents=True, exist_ok=True)

        # Extract the number at the end of model_folder
        model_folder_number = int(model_folder.split("_")[-1])
        file_name = f"attribute_prediction_features_{model_folder_number}.csv"

        if mode == "train":
            file_name = f"attribute_prediction_features_with_labels_{model_folder_number}.csv"

            # Load the challenge labels and add them to the features dataframe
            df_labels = load_dataframe(model_data_path, "challenge_label.csv")

            # Check that the number of rows align
            assert len(df_extracted_features) == len(df_labels), (
                f"The number of rows in the extracted features ({len(df_extracted_features)}) "
                f"does not match the number of labels ({len(df_labels)})."
            )
            df_extracted_features["is_train"] = df_labels.values

        save_dataframe(df=df_extracted_features, file_path=final_output_dir, file_name=file_name)


# Step 4: Attack classifier training
def run_attack_classifier_training(config: DictConfig) -> None:
    """
    Trains multiple attack classifiers to distinguish between training and synthetic data,
    and selects the best performing configuration based on evaluation metrics.

    This function orchestrates the training of various attack classifiers (XGBoost,
    CatBoost, MLP) to perform a membership inference attack. It iterates through
    different diffusion models used to generate synthetic data and all combinations
    of feature types derived from the attribute prediction task.

    The process involves:
    1.  Reading pre-computed feature files generated by the feature extraction step.
    2.  Splitting the feature files into training and testing sets.
    3.  For each diffusion model, iterating through all possible combinations of
        feature columns ('actual', 'error', 'error_ratio', 'accuracy', 'prediction').
    4.  Training each classifier type on these feature combinations.
    5.  Evaluating the classifier's performance and saving the scores (e.g., AUC, TPR at
        specific FPR) and prediction results for each configuration.
    6.  Aggregating all results into a summary CSV file, which includes a final
        metric ('final_tpr_fpr_10') representing the best TPR at 10% FPR across
        all diffusion models for a given classifier and feature set.
    7.  Logging the best-performing attack configuration based on this final metric.

    Args:
        config: Configuration object set in config.yaml.
    """
    log(INFO, "Running attack classifier training.")

    data_format, diffusion_models = (
        ("single_table", ["tabddpm", "tabsyn"])
        if config.attack_settings.single_table
        else ("multi_table", ["clavaddpm"])
    )

    # Read all the files from the attribute prediction features directory
    features_data_path = Path(config.data_paths.attribute_features_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_results: list[dict[str, Any]] = []

    for diffusion_model_name in diffusion_models:
        train_features_path = features_data_path / f"{diffusion_model_name}_black_box" / "train"

        assert train_features_path.exists() and train_features_path.is_dir(), (
            f"Directory not found: {train_features_path}. Make sure to run feature extraction first."
        )
        assert any(train_features_path.iterdir()), (
            f"Directory is empty: {train_features_path}. Make sure to run feature extraction first."
        )

        sorted_feature_files = sorted(train_features_path.glob("*.csv"))

        # Get the first 25 feature files
        train_feature_files = sorted_feature_files[:25]
        # Concatenate all the train feature files into a single dataframe
        df_train_features = pd.concat([pd.read_csv(f) for f in train_feature_files], ignore_index=True)
        train_labels = df_train_features["is_train"]
        df_train_features = df_train_features.drop(columns=["is_train"])

        test_feature_files = sorted_feature_files[25:]
        df_test_features = pd.concat([pd.read_csv(f) for f in test_feature_files], ignore_index=True)
        test_labels = df_test_features["is_train"]
        df_test_features = df_test_features.drop(columns=["is_train"])

        classifier_types = ["XGBoost", "CatBoost", "MLP"]
        column_types = ["actual", "error", "error_ratio", "accuracy", "prediction"]

        output_summary_path = Path(config.classifier_settings.results_output_path) / data_format / f"{timestamp}_train"
        output_summary_path.mkdir(parents=True, exist_ok=True)

        for classifier in classifier_types:  # XGBoost, CatBoost, MLP
            for r in range(1, len(column_types) + 1):
                for selected_columns_tuple in itertools.combinations(column_types, r):
                    # Find if a result for this combination already exists
                    columns_str = " ".join(sorted(selected_columns_tuple))
                    row = next(
                        (
                            item
                            for item in summary_results
                            if item["classifier"] == classifier and item["columns_lst"] == columns_str
                        ),
                        None,
                    )

                    if not row:
                        row = {"classifier": classifier, "columns_lst": columns_str}
                        summary_results.append(row)

                    results = train_attack_classifier(
                        classifier_type=classifier,
                        columns_list=list(selected_columns_tuple),
                        x_train=df_train_features,
                        y_train=train_labels,
                        x_test=df_test_features,
                        y_test=test_labels,
                    )

                    # Update row with scores for the current diffusion model
                    for score_name, score_value in results["scores"].items():
                        # Sanitize score_name for column header
                        col_name = score_name.lower().replace(" ", "_").replace("-", "_")
                        col_name = col_name.replace("_at_", "_").replace(".0", "")
                        row[f"{diffusion_model_name}_{col_name}"] = score_value

                    training_directory_name = f"{classifier}_" + "_".join(selected_columns_tuple)
                    training_output_path = output_summary_path / training_directory_name
                    training_output_path.mkdir(parents=True, exist_ok=True)

                    # Save prediction results
                    prediction_results_df = results["prediction_results"]
                    prediction_results_file_name = f"{diffusion_model_name}_prediction_results.csv"
                    save_dataframe(
                        df=pd.DataFrame(prediction_results_df),
                        file_path=training_output_path,
                        file_name=prediction_results_file_name,
                    )

                    # Save scores
                    scores_file_name = f"{diffusion_model_name}_results.txt"
                    with open(training_output_path / scores_file_name, "w") as f:
                        for score_name, score_value in results["scores"].items():
                            f.write(f"{score_name}: {score_value}\n")

    summary_file_name = "attack_classifier_summary.csv"
    summary_df = pd.DataFrame(summary_results)

    # Add final_tpr_fpr_10 column which is the max TPR at FPR=10% across diffusion models
    tpr_10_cols = [col for col in summary_df.columns if col.endswith("_tpr_fpr_10")]
    if tpr_10_cols:
        summary_df["final_tpr_fpr_10"] = summary_df[tpr_10_cols].max(axis=1)

    summary_df.to_csv(output_summary_path / summary_file_name, index=False)

    log(INFO, f"Saved attack classifier summary to {output_summary_path / summary_file_name}")

    summary_df.sort_values(by=["final_tpr_fpr_10"], ascending=False, inplace=True)
    best_result = summary_df.head(1)
    log(INFO, f"Best performing attack configuration:\n{best_result}")


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

    if config.pipeline.run_feature_extraction:
        run_attribute_prediction(config)

    if config.pipeline.run_attack_classifier_training:
        run_attack_classifier_training(config)


if __name__ == "__main__":
    main()
