import copy
import json
from pathlib import Path
from typing import Any
from logging import INFO
import pandas as pd

from midst_toolkit.attacks.ensemble.data_utils import load_configs, load_multi_table
from midst_toolkit.attacks.ensemble.fine_tuning_module import clava_fine_tuning
from midst_toolkit.common.logger import log
from midst_toolkit.models.clavaddpm.synthesizing import clava_synthesizing
from midst_toolkit.models.clavaddpm.train import clava_training
from midst_toolkit.models.clavaddpm.clustering import clava_clustering


def config_tabddpm(
    data_dir: Path,
    training_json_path: Path,
    final_json_path: Path,
    experiment_name: str = "attack_experiment",
    workspace_name: str = "shadow_workspace",
) -> tuple[dict, Path]:
    """
    Modifies a TabDDPM configuration JSON file with specified folder names and loads the resulting configuration.

    Args:
            data_dir: Directory containing dataset_meta.json, trans_domain.json, and trans.json files.
            training_json_path Path to the original TabDDPM training configuration JSON file.
            final_json_path: Path where the modified configuration JSON file will be saved.
            experiment_name: Name of the experiment, used to create a unique save directory.
            workspace_name: Name of the workspace, used to create a unique save directory.
    Returns:
            configs: Loaded configuration dictionary for TabDDPM.
            save_dir: Directory path where results will be saved.
    """
    # modify the config file to give the correct training data and saving directory

    with open(training_json_path, "r") as file:
        data = json.load(file)

    data["general"]["data_dir"] = str(data_dir)
    # Save dir is set by joining the workspace_dir and exp_name
    data["general"]["workspace_dir"] = str(data_dir / workspace_name)
    data["general"]["exp_name"] = experiment_name

    # save the changed to the new json file
    with open(final_json_path, "w") as file:
        json.dump(data, file, indent=4)

    log(INFO, f"DataFrame saved to {final_json_path}")

    # Set up the config
    configs, save_dir = load_configs(str(final_json_path))

    return configs, Path(save_dir)


def train_tabddpm_and_synthesize(
    train_set: pd.DataFrame,
    configs: dict[str, Any],
    save_dir: Path,
    n_synth: int = 20000,
) -> dict[str, Any]:
    """
    Train a TabDDPM model on the provided training set. Then, synthesizes data using the trained model.

    Args:
        train_set: The training dataset.
        configs: TabDDPM configuration dictionary.
        save_dir: Directory path where results will be saved.
        n_synth: Number of synthetic data points to be returned. Defaults to 20000.

    Returns:
        A dictionary containing, but not limited to, tables, trained models and synthetic data.
    """
    material = {
        "tables": {},
        "relation_order": {},
        "save_dir": save_dir,
        "all_group_lengths_prob_dicts": {},
        "models": {},
        "configs": configs,
        "synth_data": {},
    }

    # Load tables
    tables, relation_order, dataset_meta = load_multi_table(configs["general"]["data_dir"], train_df=train_set)
    material["relation_order"] = relation_order

    # Clustering on the multi-table dataset
    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, save_dir, configs)
    material["tables"] = tables
    material["all_group_lengths_prob_dicts"] = all_group_lengths_prob_dicts

    # Train models
    # TODO: check how the tables from ``clava_training`` are different from the
    # output of ``clava_clustering``.
    tables, models = clava_training(
        tables,
        relation_order,
        save_dir,
        diffusion_config=configs["diffusion"],
        classifier_config=configs["classifier"],
    )
    material["models"] = models

    if n_synth>0:
        # Determine the sample scale:
        # By default, we want the length of the final synthetic data to be len(provided_synth_data) = 20,000
        # But with a smaller scale, we can generate less synthetic data for testing purposes.
        # by default sample_scale should be 20000 / len(tables["trans"]["df"])
        # Sample scale is later multiplied by the size of training data to determine the size of synthetic data.
        sample_scale = n_synth / len(tables["trans"]["df"])
        # Generate synthetic data from scratch
        cleaned_tables, synthesizing_time_spent, matching_time_spent = clava_synthesizing(
            tables,
            relation_order,
            save_dir,
            all_group_lengths_prob_dicts,
            models,
            configs,
            sample_scale=sample_scale,
        )

        material["synth_data"] = cleaned_tables["trans"]

    return material


def fine_tune_tabddpm_and_synthesize(
    trained_models: dict[str, Any],
    new_train_set: pd.DataFrame,
    configs: dict[str, Any],
    save_dir: Path,
    fine_tuning_diffusion_iterations: int = 100,
    fine_tuning_classifier_iterations: int = 10,
    n_synth: int = 20000,
) -> dict[str, Any]:
    """
    Given a the trained models and a new training set, fine-tune the TabDDPM model.

    Args:
        trained_models: The previously trained model material.
        new_train_set: The new training dataset.
        configs: The TabDDPM configuration dictionary.
        save_dir: Directory path where results will be saved.
        fine_tuning_diffusion_iterations: Diffusion iterations for fine tuning. Defaults to 100.
        fine_tuning_classifier_iterations: Number of training iterations for the new classifier model. Defaults to 10.
        n_synth: Number of synthetic data points to be returned. Defaults to 20000.

    Returns:
        dict[str, Any]: The newly trained model material, including tables,
            relation order, models, and synthetic data.
    """
    material = {
        "tables": {},
        "relation_order": {},
        "save_dir": save_dir,
        "all_group_lengths_prob_dicts": {},
        "models": {},
        "configs": configs,
        "synth_data": {},
    }

    # Load tables
    new_tables, relation_order, dataset_meta = load_multi_table(configs["general"]["data_dir"], train_df=new_train_set)
    material["relation_order"] = relation_order

    # Clustering on the multi-table dataset
    # Original submission uses 'force_tables=True' to run the clustering even if checkpoint is found.
    new_tables, all_group_lengths_prob_dicts = clava_clustering(new_tables, relation_order, save_dir, configs)
    material["tables"] = new_tables
    material["all_group_lengths_prob_dicts"] = all_group_lengths_prob_dicts

    # Train models
    copied_models = copy.deepcopy(trained_models)
    new_models = clava_fine_tuning(
        copied_models,
        new_tables,
        relation_order,
        diffusion_config=configs["diffusion"],
        classifier_config=configs["classifier"],
        fine_tuning_diffusion_iterations = fine_tuning_diffusion_iterations,
        fine_tuning_classifier_iterations = fine_tuning_classifier_iterations,
    )
    material["new_models"] = new_models

    if n_synth>0:
        # Determine the sample scale:
        # By default, we want the length of the final synthetic data to be len(provided_synth_data) = 20,000
        # But with a smaller scale, we can generate less synthetic data for testing purposes.
        # by default sample_scale should be 20000 / len(tables["trans"]["df"])
        # Sample scale is later multiplied by the size of training data to determine the size of synthetic data.
        sample_scale = n_synth / len(new_tables["trans"]["df"])
        # Generate synthetic data from scratch
        cleaned_tables, synthesizing_time_spent, matching_time_spent = clava_synthesizing(
            new_tables,
            relation_order,
            save_dir,
            all_group_lengths_prob_dicts,
            new_models,
            configs,
            sample_scale=sample_scale,
        )

        material["synth_data"] = cleaned_tables["trans"]

    return material
