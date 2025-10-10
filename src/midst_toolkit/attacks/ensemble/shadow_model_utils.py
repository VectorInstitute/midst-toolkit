import copy
import json
import os
from logging import INFO
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from midst_toolkit.attacks.ensemble.tabddpm_fine_tuning import clava_fine_tuning
from midst_toolkit.common.logger import log
from midst_toolkit.models.clavaddpm.clustering import clava_clustering
from midst_toolkit.models.clavaddpm.data_loaders import load_multi_table
from midst_toolkit.models.clavaddpm.enumerations import Configs
from midst_toolkit.models.clavaddpm.synthesizer import clava_synthesizing
from midst_toolkit.models.clavaddpm.train import clava_training


def config_tabddpm(
    data_dir: Path,
    training_json_path: Path,
    final_json_path: Path,
    experiment_name: str = "attack_experiment",
    workspace_name: str = "shadow_workspace",
) -> tuple[Configs, Path]:
    """
    Modifies a TabDDPM configuration JSON file with the specified data directory, experiment name and workspace name,
    and loads the resulting configuration.

    Args:
            data_dir: Directory containing dataset_meta.json, trans_domain.json, and trans.json files.
            training_json_path: Path to the original TabDDPM training configuration JSON file.
            final_json_path: Path where the modified configuration JSON file will be saved.
            experiment_name: Name of the experiment, used to create a unique save directory.
            workspace_name: Name of the workspace, used to create a unique save directory.

    Returns:
            configs: Loaded configuration dictionary for TabDDPM.
            save_dir: Directory path where results will be saved.
    """
    # Modify the config file to give the correct training data and saving directory
    with open(training_json_path, "r") as file:
        config_data = json.load(file)

    config_data["general"]["data_dir"] = str(data_dir)
    # Save dir is set by joining the workspace_dir and exp_name
    config_data["general"]["workspace_dir"] = str(data_dir / workspace_name)
    config_data["general"]["exp_name"] = experiment_name

    # save the changed to the new json file
    with open(final_json_path, "w") as file:
        json.dump(config_data, file, indent=4)

    log(INFO, f"DataFrame saved to {final_json_path}")

    # Set up the config
    configs, save_dir = load_configs(str(final_json_path))

    return configs, Path(save_dir)


# TODO: This and the next function should be unified later.
def train_tabddpm_and_synthesize(
    train_set: pd.DataFrame,
    configs: Configs,
    save_dir: Path,
    synthesize: bool = True,
    sample_scale: float = 1.0,
) -> dict[str, Any]:
    """
    Train a TabDDPM model on the provided training set and optionally synthesize data using the trained models.

    Args:
        train_set: The training dataset as a pandas DataFrame.
        configs: Configuration dictionary for TabDDPM.
        save_dir: Directory path where models and results will be saved.
        synthesize: Flag indicating whether to generate synthetic data after training. Defaults to True.
        sample_scale: Factor to scale the number of synthesized samples relative to the training set size.
            Defaults to 1.0.

    Returns:
        A dictionary containing tables, trained models, synthetic data, and other relevant information.
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
    tables, relation_order, dataset_meta = load_multi_table(
        Path(configs["general"]["data_dir"]), train_data={"trans": train_set}
    )
    material["relation_order"] = relation_order

    # Clustering on the multi-table dataset
    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, save_dir, configs)
    material["tables"] = tables
    material["all_group_lengths_prob_dicts"] = all_group_lengths_prob_dicts

    # Train models
    tables, models = clava_training(
        tables,
        relation_order,
        save_dir,
        diffusion_config=configs["diffusion"],
        classifier_config=configs["classifier"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    material["models"] = models

    if synthesize:
        # By default, we want the length of the final synthetic data to be ``len(provided_synth_data) = 20,000``
        # But with a smaller scale, we can generate less synthetic data for debugging purposes.
        # Attack's default sample_scale is set to ``20000 / len(tables["trans"]["df"])`` to
        # generate 20,000 samples regardless
        # of the training data size.
        # Sample scale is later multiplied by the size of training data (no id) to determine
        # the size of synthetic data.
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
    trained_models: dict[tuple[str, str], dict[str, Any]],
    new_train_set: pd.DataFrame,
    configs: Configs,
    save_dir: Path,
    fine_tuning_diffusion_iterations: int = 100,
    fine_tuning_classifier_iterations: int = 10,
    synthesize: bool = True,
    sample_scale: float = 1.0,
) -> dict[str, Any]:
    """
    Given the trained models and a new training set, fine-tune the TabDDPM models.
    If ``synthesize`` is True, synthesizes data using the fine-tuned models. Number of
    synthesized data points is determined by the ``classifier_scale`` parameter in training ``configs``.

    Args:
        trained_models: The previously trained model material.
        new_train_set: The new training dataset for fine-tuning.
        configs: Configuration dictionary for TabDDPM.
        save_dir:  Directory path where models and results will be saved.
        fine_tuning_diffusion_iterations: Diffusion iterations for fine tuning. Defaults to 100.
        fine_tuning_classifier_iterations: Number of training iterations for the new classifier model.
            Defaults to 10.
        synthesize: Flag indicating whether to generate synthetic data after training. Defaults to True.
        sample_scale: Factor to scale the number of synthesized samples relative to the training set size.
            Defaults to 1.0.

    Returns:
        dict[str, Any]: The newly trained model material, including tables,
            relation order, models, and synthetic data.
    """
    material = {
        "tables": {},
        "relation_order": {},
        "save_dir": save_dir,
        "all_group_lengths_prob_dicts": {},
        "new_models": {},
        "configs": configs,
        "synth_data": {},
    }

    # Load tables
    new_tables, relation_order, dataset_meta = load_multi_table(
        Path(configs["general"]["data_dir"]),
        train_data={"trans": new_train_set},
    )
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
        fine_tuning_diffusion_iterations=fine_tuning_diffusion_iterations,
        fine_tuning_classifier_iterations=fine_tuning_classifier_iterations,
    )
    material["new_models"] = new_models

    if synthesize:
        # By default, we want the length of the final synthetic data to be ``len(provided_synth_data) = 20,000``
        # But with a smaller scale, we can generate less synthetic data for debugging purposes.
        # Ensemble Attack's default sample_scale is ``20000 / len(tables["trans"]["df"])`` to generate 20,000 samples
        # regardless of the train data size.
        # Sample scale is later multiplied by the size of training data to determine the size of synthetic data.
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


# TODO: The following function is directly copied from the midst reference code since
# I need it to run the attack code, but, it should probably be moved to somewhere else.
def load_configs(config_path: str) -> tuple[Configs, Path]:
    """
    Load configuration from a JSON file and set up necessary directories.

    Args:
        config_path: Path to the configuration JSON file.

    Returns:
        configs: Loaded configuration dictionary.
        save_dir: Directory path where results will be saved.
    """
    with open(config_path, "r") as file:
        configs = json.load(file)

    # Following directories are created to save the models and intermediate results.
    save_dir = os.path.join(configs["general"]["workspace_dir"], configs["general"]["exp_name"])
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "before_matching"), exist_ok=True)

    with open(os.path.join(save_dir, "args"), "w") as file:
        json.dump(configs, file, indent=4)

    return configs, Path(save_dir)
