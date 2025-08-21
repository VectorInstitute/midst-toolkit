import copy
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

# TODO: The following unrecognized modules should be imported from our own library once these
# functions are added. Right now the following modules are implemented in
# "midst_models/single_table_TabDDPM/complex_pipeline.py"
from MIDSTModels.midst_models.single_table_TabDDPM.complex_pipeline import (
    clava_synthesizing,
    load_configs,
)
from MIDSTModels.midst_models.single_table_TabDDPM.pipeline_modules import (
    load_multi_table,
)

from midst_toolkit.models.clavaddpm.fine_tuning_module import clava_fine_tuning
from midst_toolkit.models.clavaddpm.model import clava_clustering, clava_training


def config_tabddpm(
    data_dir: Path,
    json_path: Path | None = None,
    final_json_path: Path | None = None,
    diffusion_layers: list[int] | None = None,
    diffusion_iterations: int = 10000,
    classifier_layers: list[int] | None = None,
    classifier_dim_t: int = 16,
    classifier_iterations: int = 1000,
) -> tuple[dict, Path]:
    """
    Modifies a TabDDPM configuration JSON file with specified parameters and loads the resulting configuration.

    Args:
            data_dir (Path): Directory containing dataset_meta.json, trans_domain.json, and trans.json files.
            json_path (Path | None): Path to the input JSON configuration file. If None, uses 'trans.json' in data_dir.
            final_json_path (Path | None): Path to save the modified JSON configuration file.
            diffusion_layers (list[int] | None): List specifying the number of units in each diffusion model layer.
            diffusion_iterations (int): Number of training iterations for the diffusion model.
            classifier_layers (list[int] | None): List specifying the number of units in each classifier model layer.
            classifier_dim_t (int): Dimension of the classifier's time embedding.
            classifier_iterations (int): Number of training iterations for the classifier model.

    Returns:
            tuple[dict, str]:
                - configs (dict): Loaded configuration dictionary for TabDDPM.
                - save_dir (str): Directory path where results will be saved.
    """
    if diffusion_layers is None:
        diffusion_layers = [32, 64, 64, 64, 64, 32]
    if classifier_layers is None:
        classifier_layers = [16, 32, 64, 128, 64, 32, 16]

    # modify the config file to give the correct training data and saving directory
    temp_json_file_path = json_path if json_path is not None else data_dir / "trans.json"

    with open(temp_json_file_path, "r") as file:
        data = json.load(file)
    data["general"]["data_dir"] = data_dir
    data["general"]["exp_name"] = "tmp"
    data["general"]["workspace_dir"] = data_dir / "tmp_workspace"

    # modify the model parameters for smaller sets
    data["diffusion"]["d_layers"] = diffusion_layers
    data["diffusion"]["iterations"] = diffusion_iterations
    data["classifier"]["d_layers"] = classifier_layers
    data["classifier"]["dim_t"] = classifier_dim_t
    data["classifier"]["iterations"] = classifier_iterations

    # save the changed to the new json file
    final_json_file_path = final_json_path if final_json_path is not None else temp_json_file_path

    with open(final_json_file_path, "w") as file:
        json.dump(data, file, indent=4)

    logging.info(f"DataFrame saved to {final_json_file_path}")

    # Set up the config
    configs, save_dir = load_configs(final_json_file_path)

    return configs, Path(save_dir)


def train_tabddpm(
    train_set: pd.DataFrame,
    configs: dict[str, Any],
    save_dir: Path,
) -> dict[str, Any]:
    """
    Train a TabDDPM model on the provided training set.

    Args:
        train_set (pd.DataFrame): The training dataset as a pandas DataFrame.
        configs (dict): TabDDPM configuration dictionary.
        save_dir (Path): Directory path where results will be saved.

    Returns:
        dict[str, Any]: A dictionary containing tables, trained models and synthetic data.
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
    models = clava_training(tables, relation_order, save_dir, configs)
    material["models"] = models

    # Determine the sample scale
    # We want the final synthetic data = len(provided_synth_data) = 20,000
    sample_scale = 20000 / len(tables["trans"]["df"])

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


def fine_tune_tabddpm(
    trained_models: dict[str, Any],
    new_train_set: pd.DataFrame,
    configs: dict[str, Any],
    save_dir: Path,
    new_diffusion_iterations: int = 100,
    new_classifier_iterations: int = 10,
    n_synth: int = 20000,
) -> dict[str, Any]:
    """
    Given a the trained models and a new training set, fine-tune the TabDDPM model.

    Args:
        trained_models (dict[str, any]): The previously trained model material.
        new_train_set (pd.DataFrame): The new training dataset as a pandas DataFrame.
        configs (dict[str, Any]): The TabDDPM configuration dictionary.
        save_dir (Path): Directory path where results will be saved.
        new_diffusion_iterations (int): Diffusion iterations for fine tuning. Defaults to 100.
        new_classifier_iterations (int): Number of training iterations for the new classifier model. Defaults to 10.
        n_synth (int, optional): Number of synthetic data points to be returned. Defaults to 20000.

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
        configs,
        new_diffusion_iterations,
        new_classifier_iterations,
    )
    material["new_models"] = new_models

    # Determine the sample scale
    # We want the final synthetic data = len(provided_synth_data) = 20,000
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
