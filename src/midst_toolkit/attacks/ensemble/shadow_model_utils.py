import copy
import json
import os
from dataclasses import dataclass
from logging import INFO
from pathlib import Path

import pandas as pd
import torch

from midst_toolkit.attacks.ensemble.clavaddpm_fine_tuning import clava_fine_tuning
from midst_toolkit.common.config import TrainingConfig
from midst_toolkit.common.logger import log
from midst_toolkit.models.clavaddpm.clustering import clava_clustering
from midst_toolkit.models.clavaddpm.data_loaders import load_tables
from midst_toolkit.models.clavaddpm.enumerations import (
    GroupLengthsProbDicts,
    ModelArtifacts,
    Relation,
    RelationOrder,
    Tables,
)
from midst_toolkit.models.clavaddpm.synthesizer import clava_synthesizing
from midst_toolkit.models.clavaddpm.train import clava_training


@dataclass
class TrainingResult:
    save_dir: Path
    configs: TrainingConfig
    tables: Tables
    relation_order: RelationOrder
    all_group_lengths_probabilities: GroupLengthsProbDicts
    models: dict[Relation, ModelArtifacts]
    synthetic_data: pd.DataFrame | None = None


def save_additional_tabddpm_config(
    data_dir: Path,
    training_config_json_path: Path,
    final_config_json_path: Path,
    experiment_name: str = "attack_experiment",
    workspace_name: str = "shadow_workspace",
) -> tuple[TrainingConfig, Path]:
    """
    Modifies a TabDDPM configuration JSON file with the specified data directory, experiment name and workspace name,
    and loads the resulting configuration.

    Args:
            data_dir: Directory containing dataset_meta.json, trans_domain.json, and trans.json files.
            training_config_json_path: Path to the original TabDDPM training configuration JSON file.
            final_config_json_path: Path where the modified configuration JSON file will be saved.
            experiment_name: Name of the experiment, used to create a unique save directory.
            workspace_name: Name of the workspace, used to create a unique save directory.

    Returns:
            configs: Loaded configuration dictionary for TabDDPM.
            save_dir: Directory path where results will be saved.
    """
    # Modify the config file to give the correct training data and saving directory
    with open(training_config_json_path, "r") as file:
        configs = TrainingConfig(**json.load(file))

    configs.general.data_dir = data_dir
    # Save dir is set by joining the workspace_dir and exp_name
    configs.general.workspace_dir = data_dir / workspace_name
    configs.general.exp_name = experiment_name

    # save the changed to the new json file
    with open(final_config_json_path, "w") as file:
        json.dump(configs.model_dump(mode="json"), file, indent=4)

    log(INFO, f"Config saved to {final_config_json_path}")

    # Set up the config
    save_dir = setup_save_dir(configs)

    return configs, save_dir


# TODO: This and the next function should be unified later.
def train_tabddpm_and_synthesize(
    train_set: pd.DataFrame,
    configs: TrainingConfig,
    save_dir: Path,
    synthesize: bool = True,
    sample_scale: float = 1.0,
) -> TrainingResult:
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
        A dataclass TrainingResult object containing:
            - save_dir: Directory where results are saved.
            - configs: Configuration dictionary used for training.
            - tables: Loaded tables after clustering.
            - relation_order: Relation order of the tables.
            - all_group_lengths_probabilities: Group lengths probability dictionaries.
            - models: The trained models.
            - synthetic_data: The synthesized data as a pandas DataFrame, if synthesis was performed,
              otherwise, None.
    """
    # Load tables
    tables, relation_order, _ = load_tables(configs.general.data_dir, train_data={"trans": train_set})

    # Clustering on the multi-table dataset
    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, save_dir, configs.clustering)

    # Train models
    tables, models = clava_training(
        tables,
        relation_order,
        save_dir,
        diffusion_config=configs.diffusion,
        classifier_config=configs.classifier,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    result = TrainingResult(
        save_dir=save_dir,
        configs=configs,
        tables=tables,
        relation_order=relation_order,
        all_group_lengths_probabilities=all_group_lengths_prob_dicts,
        models=models,
    )

    if synthesize:
        # By default, we want the length of the final synthetic data to be ``len(provided_synth_data) = 20,000``
        # But with a smaller scale, we can generate less synthetic data for debugging purposes.
        # Attack's default sample_scale is set to ``20000 / len(tables["trans"]["df"])`` to
        # generate 20,000 samples regardless
        # of the training data size.
        # Sample scale is later multiplied by the size of training data (no id) to determine
        # the size of synthetic data.
        cleaned_tables, _, _ = clava_synthesizing(
            tables,
            relation_order,
            save_dir,
            all_group_lengths_prob_dicts,
            models,
            configs.general,
            configs.sampling,
            configs.matching,
            sample_scale=sample_scale,
        )

        result.synthetic_data = cleaned_tables["trans"]

    return result


def fine_tune_tabddpm_and_synthesize(
    trained_models: dict[Relation, ModelArtifacts],
    fine_tune_set: pd.DataFrame,
    configs: TrainingConfig,
    save_dir: Path,
    fine_tuning_diffusion_iterations: int = 100,
    fine_tuning_classifier_iterations: int = 10,
    synthesize: bool = True,
    sample_scale: float = 1.0,
) -> TrainingResult:
    """
    Given the trained models and a new training set, fine-tune the TabDDPM models.
    If ``synthesize`` is True, synthesizes data using the fine-tuned models. Number of
    synthesized data points is determined by the ``classifier_scale`` parameter in training ``configs``.

    Args:
        trained_models: The previously trained model material.
        fine_tune_set: The new training dataset for fine-tuning.
        configs: Configuration dictionary for TabDDPM.
        save_dir:  Directory path where models and results will be saved.
        fine_tuning_diffusion_iterations: Diffusion iterations for fine tuning. Defaults to 100.
        fine_tuning_classifier_iterations: Number of training iterations for the new classifier model.
            Defaults to 10.
        synthesize: Flag indicating whether to generate synthetic data after training. Defaults to True.
        sample_scale: Factor to scale the number of synthesized samples relative to the training set size.
            Defaults to 1.0.

    Returns:
        A dataclass TrainingResult object containing:
            - save_dir: Directory where results are saved.
            - configs: Configuration dictionary used for training.
            - tables: Loaded tables after clustering.
            - relation_order: Relation order of the tables.
            - all_group_lengths_probabilities: Group lengths probability dictionaries.
            - models: The trained models.
            - synthetic_data: The synthesized data as a pandas DataFrame, if synthesis was performed,
              otherwise, None.
    """
    # Load tables
    new_tables, relation_order, _ = load_tables(configs.general.data_dir, train_data={"trans": fine_tune_set})

    # Clustering on the multi-table dataset
    # Original submission uses 'force_tables=True' to run the clustering even if checkpoint is found.
    new_tables, all_group_lengths_prob_dicts = clava_clustering(
        new_tables, relation_order, save_dir, configs.clustering
    )

    # Train models
    copied_models = copy.deepcopy(trained_models)
    new_models = clava_fine_tuning(
        copied_models,
        new_tables,
        relation_order,
        diffusion_config=configs.diffusion,
        classifier_config=configs.classifier,
        fine_tuning_diffusion_iterations=fine_tuning_diffusion_iterations,
        fine_tuning_classifier_iterations=fine_tuning_classifier_iterations,
    )
    result = TrainingResult(
        save_dir=save_dir,
        configs=configs,
        tables=new_tables,
        relation_order=relation_order,
        all_group_lengths_probabilities=all_group_lengths_prob_dicts,
        models=new_models,
    )

    if synthesize:
        # By default, we want the length of the final synthetic data to be ``len(provided_synth_data) = 20,000``
        # But with a smaller scale, we can generate less synthetic data for debugging purposes.
        # Ensemble Attack's default sample_scale is ``20000 / len(tables["trans"]["df"])`` to generate 20,000 samples
        # regardless of the train data size.
        # Sample scale is later multiplied by the size of training data to determine the size of synthetic data.
        cleaned_tables, _, _ = clava_synthesizing(
            new_tables,
            relation_order,
            save_dir,
            all_group_lengths_prob_dicts,
            new_models,
            configs.general,
            configs.sampling,
            configs.matching,
            sample_scale=sample_scale,
        )

        result.synthetic_data = cleaned_tables["trans"]

    return result


# TODO: The following function is directly copied from the midst reference code since
# I need it to run the attack code, but, it should probably be moved to somewhere else
# as it is an essential part of a working TabDDPM training pipeline.
def setup_save_dir(configs: TrainingConfig) -> Path:
    """
    Set up the directories where the models and intermediate results will be saved.

    The following directories are created:
        - save_dir -> configs.general.workspace_dir/configs.general.exp_name
        - save_dir/models
        - save_dir/before_matching

    Additionally, a json file with the configuration settings is saved to ``save_dir/args``.

    Args:
        configs: Configuration settings.

    Returns:
        save_dir: Directory path where results will be saved.
    """
    # Following directories are created to save the models and intermediate results.
    save_dir = configs.general.workspace_dir / configs.general.exp_name
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir / "models", exist_ok=True)
    os.makedirs(save_dir / "before_matching", exist_ok=True)

    with open(save_dir / "args", "w") as file:
        json.dump(configs.model_dump(mode="json"), file, indent=4)

    return save_dir
