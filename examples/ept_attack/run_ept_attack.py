"""
This file is an incomplete example script for running the EPT-MIA Attack on MIDST challenge
provided resources and data.
Overall workflow and decisions are taken with from the Cyber@BGU team's attack implementation at
https://github.com/eyalgerman/MIA-EPT.

"""

import itertools
import json
import pickle
from collections import defaultdict
from datetime import datetime
from logging import INFO
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from examples.common.utils import directory_checks, iterate_model_folders
from midst_toolkit.attacks.ensemble.data_utils import load_dataframe, save_dataframe
from midst_toolkit.attacks.ept.classification import ClassifierType, train_attack_classifier
from midst_toolkit.attacks.ept.feature_extraction import extract_features
from midst_toolkit.common.logger import log
from midst_toolkit.common.random import set_all_random_seeds


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
    directory_checks(input_data_path)

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


def _summarize_and_save_training_results(
    summary_results: dict, output_summary_path: Path, summary_file_name: str
) -> pd.DataFrame:
    """
    Processes summary results, saves them to a CSV, and returns the summary DataFrame.

    Args:
        summary_results: A dictionary containing the summary results.
        output_summary_path: The path where the summary CSV will be saved.
        summary_file_name: The name of the summary CSV file.

    Returns:
        A pandas DataFrame containing the summarized results.
    """
    processed_results = []
    for (classifier, columns_lst), model_scores in summary_results.items():
        row: dict[str, str | float] = {"classifier": classifier, "column_types": columns_lst}
        for diffusion_model_name, scores in model_scores:
            for score_name, score_value in scores.items():
                col_name = (
                    score_name.lower().replace(" ", "_").replace("-", "_").replace("_at_", "_").replace(".0", "")
                )
                row[f"{diffusion_model_name}_{col_name}"] = score_value
        processed_results.append(row)

    summary_df = pd.DataFrame(processed_results)
    tpr_10_cols = [col for col in summary_df.columns if col.endswith("_tpr_fpr_10")]
    if tpr_10_cols:
        summary_df["final_tpr_fpr_10"] = summary_df[tpr_10_cols].max(axis=1)

    summary_df.to_csv(output_summary_path / summary_file_name, index=False)
    log(INFO, f"Saved attack classifier summary to {output_summary_path / summary_file_name}")
    return summary_df


def _train_and_save_best_attack_classifier(
    config: DictConfig, best_result: pd.DataFrame, diffusion_model_name: str, model_save_path: Path
) -> None:
    """
    Trains and saves the best attack classifier based on the summary DataFrame.

    Args:
        config: Configuration object set in config.yaml.
        best_result: DataFrame containing the best attack configuration (classifier and column types).
        diffusion_model_name: Name of the diffusion model  (e.g., 'tabddpm', 'tabsyn', 'clavaddpm').
        model_save_path: Path where the trained model will be saved.
    """
    # Train and save the best attack classifier
    best_classifier_name = best_result["classifier"].iloc[0]
    best_column_types_str = best_result["column_types"].iloc[0]
    best_column_types = best_column_types_str.split(" ")

    log(
        INFO,
        f"Training final attack model for {diffusion_model_name} with classifier: {best_classifier_name} and features: {best_column_types}",
    )

    train_features_data_path = (
        Path(config.data_paths.attribute_features_path) / f"{diffusion_model_name}_black_box" / "train"
    )

    # Concatenate all train features and labels for final training
    train_feature_files = train_features_data_path.glob("*.csv")
    df_train_features = pd.concat([pd.read_csv(f) for f in train_feature_files], ignore_index=True)
    train_labels = df_train_features["is_train"]
    df_train_features = df_train_features.drop(columns=["is_train"])

    # Train the final model
    final_model_results = train_attack_classifier(
        classifier_type=ClassifierType(best_classifier_name),
        column_types=best_column_types,
        x_train=df_train_features,
        y_train=train_labels,
        x_test=None,  # No test set, training on all available data
        y_test=None,
    )

    final_model = final_model_results["trained_model"]

    model_save_path = Path(model_save_path) / f"{diffusion_model_name}_best_attack_classifier.pkl"

    with open(model_save_path, "wb") as file:
        pickle.dump(final_model, file)

    log(INFO, f"Saved the best attack model to {model_save_path}")


# Step 4: Attack classifier training
def run_attack_classifier_training(config: DictConfig) -> None:
    """
    Trains multiple attack classifiers to distinguish between training and
    non-training data, and selects the best performing configuration based
    on evaluation metrics.

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
    8.  Training and saving the best attack classifier using all available training data.

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

    # An example of summary_results structure:
    # {
    #   ('XGBoost', 'actual error'): [
    #       ('tabddpm', {'AUC': 0.85, 'TPR at FPR=10%': 0.75, ...}),
    #       ('tabsyn', {'AUC': 0.80, 'TPR at FPR=10%': 0.70, ...}),
    #   ],
    #   ('CatBoost', 'accuracy prediction'): [
    #       ('tabddpm', {'AUC': 0.82, 'TPR at FPR=10%': 0.72, ...}),
    #       ('tabsyn', {'AUC': 0.78, 'TPR at FPR=10%    ': 0.68, ...}),
    #   ],
    #   ...
    # }

    # TODO: Move this part of code to a separate function (hyper-parameter tuning)
    # TODO: Move some of the code to midst_toolkit.attacks.ept.classification module

    summary_results: dict[tuple[str, str], list[tuple[str, dict[str, float]]]] = defaultdict(list)

    for diffusion_model_name in diffusion_models:
        train_features_path = features_data_path / f"{diffusion_model_name}_black_box" / "train"

        directory_checks(train_features_path, "Make sure to run feature extraction first.")

        sorted_feature_files = sorted(train_features_path.glob("*.csv"))
        split_index = len(sorted_feature_files) * 5 // 6

        # Get the first 25 feature files
        train_feature_files = sorted_feature_files[:split_index]
        # Concatenate all the train feature files into a single dataframe
        df_train_features = pd.concat([pd.read_csv(f) for f in train_feature_files], ignore_index=True)
        train_labels = df_train_features["is_train"]
        df_train_features = df_train_features.drop(columns=["is_train"])

        test_feature_files = sorted_feature_files[split_index:]
        df_test_features = pd.concat([pd.read_csv(f) for f in test_feature_files], ignore_index=True)
        test_labels = df_test_features["is_train"]
        df_test_features = df_test_features.drop(columns=["is_train"])

        classifier_types = ["XGBoost", "CatBoost", "MLP"]
        column_types = ["actual", "error", "error_ratio", "accuracy", "prediction"]

        output_summary_path = Path(config.classifier_settings.results_output_path) / data_format / f"{timestamp}_train"
        output_summary_path.mkdir(parents=True, exist_ok=True)

        for classifier in classifier_types:
            for r in range(1, len(column_types) + 1):
                for selected_column_types_tuple in itertools.combinations(column_types, r):
                    columns_str = " ".join(sorted(selected_column_types_tuple))
                    result_key = (classifier, columns_str)

                    classifier_type = ClassifierType(classifier)

                    results = train_attack_classifier(
                        classifier_type=classifier_type,
                        column_types=list(selected_column_types_tuple),
                        x_train=df_train_features,
                        y_train=train_labels,
                        x_test=df_test_features,
                        y_test=test_labels,
                    )

                    # Store raw scores for the current diffusion model
                    summary_results[result_key].append((diffusion_model_name, results["scores"]))

                    training_directory_name = f"{classifier}_{'_'.join(selected_column_types_tuple)}"
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

    summary_df = _summarize_and_save_training_results(
        summary_results, output_summary_path, "attack_classifier_summary.csv"
    )

    if data_format == "single_table":
        # For single-table data, focus on tabddpm results
        summary_df.sort_values(by=["tabddpm_tpr_fpr_10"], ascending=False, inplace=True)
    else:
        # For multi-table data, get the clavaddpm results
        summary_df.sort_values(by=["clavaddpm_tpr_fpr_10"], ascending=False, inplace=True)

    best_result = summary_df.head(1)
    log(INFO, f"Best performing attack configuration:\n{best_result}")

    for diffusion_model_name in diffusion_models:
        model_save_path = Path(config.classifier_settings.results_output_path) / data_format
        _train_and_save_best_attack_classifier(config, best_result, diffusion_model_name, model_save_path)


def run_inference(config: DictConfig) -> None:
    """
    Runs inference using the trained attack classifier on the challenge data.

    Args:
        config: Configuration object set in config.yaml.
    """
    log(INFO, "Running inference with the trained attack classifier.")

    data_format, diffusion_models = (
        ("single_table", ["tabddpm", "tabsyn"])
        if config.attack_settings.single_table
        else ("multi_table", ["clavaddpm"])
    )

    for diffusion_model_name in diffusion_models:
        # Load the trained attack classifier
        model_path = (
            Path(config.classifier_settings.results_output_path)
            / data_format
            / f"{diffusion_model_name}_best_attack_classifier.pkl"
        )

        with open(model_path, "rb") as file:
            trained_model = pickle.load(file)

        # Load new feature data for inference
        features_data_path = Path(config.data_paths.attribute_features_path)
        inference_features_path = features_data_path / f"{diffusion_model_name}_black_box" / "final"

        directory_checks(inference_features_path, "Make sure to run feature extraction on final data first.")

        challenge_feature_files = inference_features_path.glob("*.csv")

        df_inference_features = pd.concat([pd.read_csv(f) for f in challenge_feature_files], ignore_index=True)

        predictions = trained_model.predict(df_inference_features)

        # Save inference results
        inference_output_path = Path(config.data_paths.inference_results_path)
        inference_output_path.mkdir(parents=True, exist_ok=True)

        inference_results_file_name = f"{diffusion_model_name}_attack_inference_results.csv"

        save_dataframe(
            df=pd.DataFrame({"prediction": predictions}),
            file_path=inference_output_path,
            file_name=inference_results_file_name,
        )

        log(INFO, f"Saved inference results to {inference_output_path / inference_results_file_name}")

        # TODO: Implement evaluation of inference results using the challenge labels
        # _evaluate_inference_results(predictions, diffusion_model_name)


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

    if config.random_seed is not None:
        set_all_random_seeds(seed=config.random_seed)
        log(INFO, f"Training phase random seed set to {config.random_seed}.")

    if config.attack_settings.single_table:
        log(INFO, "Data: Single-table.")
    else:
        log(INFO, "Data: Multi-table.")

    # TODO: Implement potential data preprocessing step.

    if config.pipeline.run_feature_extraction:
        run_attribute_prediction(config)

    if config.pipeline.run_attack_classifier_training:
        run_attack_classifier_training(config)

    if config.pipeline.run_inference:
        run_inference(config)


if __name__ == "__main__":
    main()
