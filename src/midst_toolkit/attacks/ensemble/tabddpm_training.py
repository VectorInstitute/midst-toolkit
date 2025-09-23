import copy
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from midst_toolkit.attacks.ensemble.data_utils import load_configs, load_multi_table
from midst_toolkit.models.clavaddpm.fine_tuning_module import clava_fine_tuning

# TODO: The following unrecognized modules should be imported from our own library once these
# functions are added. Right now the following modules are implemented in
# "midst_models/single_table_TabDDPM/complex_pipeline.py"
from midst_toolkit.models.clavaddpm.synthesizing import clava_synthesizing
from midst_toolkit.models.clavaddpm.train import clava_clustering, clava_training


def config_tabddpm(
    data_dir: Path,
    training_json_path: Path,
    final_json_path: Path,
    experiment_name: str = "tmp",
    workspace_name: str = "shadow_workspace",
) -> tuple[dict, Path]:
    """
    Modifies a TabDDPM configuration JSON file with specified parameters and loads the resulting configuration.

    Args:
            data_dir (Path): Directory containing dataset_meta.json, trans_domain.json, and trans.json files.
            # TODO: fix docstring
    Returns:
            tuple[dict, str]:
                - configs (dict): Loaded configuration dictionary for TabDDPM.
                - save_dir (str): Directory path where results will be saved.
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

    logging.info(f"DataFrame saved to {final_json_path}")

    # Set up the config
    configs, save_dir = load_configs(str(final_json_path))

    return configs, Path(save_dir)


def train_tabddpm_and_synthesize(
    train_set: pd.DataFrame,
    configs: dict[str, Any],
    save_dir: Path,
    synthesize: bool = True,
) -> dict[str, Any]:
    """
    Train a TabDDPM model on the provided training set. The, synthesize data using the trained model.

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

    if synthesize:
        # Determine the sample scale
        # We want the final synthetic data = len(provided_synth_data) = 20,000
        # TODO: fix sample scale to sample based on the requested number of synth data points
        # sample_scale = 20000 / len(tables["trans"]["df"])
        sample_scale = 1.0

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
        fine_tuning_diffusion_iterations,
        fine_tuning_classifier_iterations,
    )
    material["new_models"] = new_models

    # Determine the sample scale
    # We want the final synthetic data = len(provided_synth_data) = 20,000
    # TODO: fix next line
    # sample_scale = n_synth / len(new_tables["trans"]["df"])
    sample_scale = 0.001

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
