"""
Module containing the base classes and implementations for the Ensemble Attack model runner and training result.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from logging import INFO
import copy
import json

import pandas as pd
from pydantic import BaseModel, ConfigDict
from sdv.single_table import CTGANSynthesizer  # type: ignore[import-untyped]
from sdv.metadata import SingleTableMetadata  # type: ignore[import-untyped]

from midst_toolkit.common.config import ClavaDDPMTrainingConfig, CTGANTrainingConfig, TrainingConfig
from midst_toolkit.models.clavaddpm.data_loaders import Tables, load_tables
from midst_toolkit.models.clavaddpm.enumerations import GroupLengthsProbDicts, Relation, RelationOrder
from midst_toolkit.models.clavaddpm.train import ClavaDDPMModelArtifacts, CTGANModelArtifacts
from midst_toolkit.models.clavaddpm.clustering import clava_clustering
from midst_toolkit.models.clavaddpm.train import clava_training
from midst_toolkit.models.clavaddpm.synthesizer import clava_synthesizing
from midst_toolkit.common.variables import DEVICE
from midst_toolkit.common.logger import log
from midst_toolkit.attacks.ensemble.clavaddpm_fine_tuning import clava_fine_tuning




# Base Classes
class EnsembleAttackTrainingConfig(TrainingConfig):
    number_of_points_to_synthesize: int = 20000

class EnsembleAttackTrainingResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    save_dir: Path
    configs: EnsembleAttackTrainingConfig
    models: Any
    synthetic_data: pd.DataFrame | None = None


class EnsembleAttackModelRunner(ABC):
    def __init__(self, training_config: EnsembleAttackTrainingConfig):
        """
        Initialize the ensemble attackmodel runner with a training config.

        Args:
            training_config: The training config for the ensemble attack model.
        """
        self.training_config = training_config

    @abstractmethod
    def train_or_fine_tune_and_synthesize(
        self,
        dataset: pd.DataFrame,
        synthesize: bool = True,
        trained_model: EnsembleAttackTrainingResult | None = None,
    ) -> EnsembleAttackTrainingResult:
        """
        Train or fine tune a model and synthesize data.

        Args:
            dataset: The dataset to train or fine tune the model on.
            synthesize: Whether to synthesize data after training.
            trained_model: The model to fine tune. If None, a new model should be trained.
                Optional, default is None.

        Returns:
            An instance of `EnsembleAttackTrainingResult` containing the training results.
        """
        raise NotImplementedError("Subclasses must implement this method.")


# TabDDPM/ClavaDDPM implementation
class EnsembleAttackTabDDPMTrainingConfig(ClavaDDPMTrainingConfig, EnsembleAttackTrainingConfig):
    fine_tuning_diffusion_iterations: int = 100
    fine_tuning_classifier_iterations: int = 10


class TabDDPMTrainingResult(EnsembleAttackTrainingResult):
    configs: EnsembleAttackTabDDPMTrainingConfig
    models: dict[Relation, ClavaDDPMModelArtifacts]
    tables: Tables
    relation_order: RelationOrder
    all_group_lengths_probabilities: GroupLengthsProbDicts


class EnsembleAttackTabDDPMModelRunner(EnsembleAttackModelRunner):
    def train_or_fine_tune_and_synthesize(
        self,
        dataset: pd.DataFrame,
        synthesize: bool = True,
        trained_model: EnsembleAttackTrainingResult | None = None,
    ) -> TabDDPMTrainingResult:
        """
        Train or fine tune a TabDDPM model on the provided training set and optionally synthesize
        data using the trained/fine-tuned models.

        Args:
            dataset: The training dataset as a pandas DataFrame.
            synthesize: Flag indicating whether to generate synthetic data after training. Defaults to True.
            trained_model: The model to fine tune. If None, a new model should be trained.
                Optional, default is None.

        Returns:
            A dataclass TabDDPMTrainingResult object containing:
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
        tables, relation_order, _ = load_tables(self.training_config.general.data_dir, train_data={"trans": dataset})

        save_dir = self.training_config.general.workspace_dir / self.training_config.general.exp_name

        # Clustering on the multi-table dataset
        tables, all_group_lengths_prob_dicts = clava_clustering(
            tables,
            relation_order,
            save_dir,
            self.training_config.clustering,
        )

        if trained_model is None:
            # Train models
            models = clava_training(
                tables,
                relation_order,
                save_dir,
                diffusion_config=self.training_config.diffusion,
                classifier_config=self.training_config.classifier,
                device=DEVICE,
            )

        else:
            # Fine-tune models
            copied_models = copy.deepcopy(trained_model.models)
            models = clava_fine_tuning(
                copied_models,
                tables,
                relation_order,
                diffusion_config=self.training_config.diffusion,
                classifier_config=self.training_config.classifier,
                fine_tuning_diffusion_iterations=self.training_config.fine_tuning_diffusion_iterations,
                fine_tuning_classifier_iterations=self.training_config.fine_tuning_classifier_iterations,
            )

        result = TabDDPMTrainingResult(
            save_dir=save_dir,
            configs=self.training_config,
            tables=tables,
            relation_order=relation_order,
            all_group_lengths_probabilities=all_group_lengths_prob_dicts,
            models=models,
        )

        if synthesize:
            # By default, Ensemble attack generates a synthetic data of length ``20,000``.
            # Attack's default sample_scale is set to ``20000 / len(tables["trans"]["df"])`` to
            # generate 20,000 samples regardless of the training data size. But we control the
            # synthetic data size directly here with ``number_of_points_to_synthesize``.
            # ``sample_scale`` is later multiplied by the size of training data (no id) to determine
            # the size of synthetic data.
            assert len(tables["trans"].data) > 0, "Cannot synthesize: training data is empty"
            sample_scale = self.training_config.number_of_points_to_synthesize / len(tables["trans"].data)
            cleaned_tables, _, _ = clava_synthesizing(
                tables,
                relation_order,
                save_dir,
                models,
                self.training_config.general,
                self.training_config.sampling,
                self.training_config.matching,
                all_group_lengths_prob_dicts,
                sample_scale=sample_scale,
            )

            result.synthetic_data = cleaned_tables["trans"]

        return result


# CTGAN implementation
class EnsembleAttackCTGANTrainingConfig(CTGANTrainingConfig, EnsembleAttackTrainingConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: SingleTableMetadata = None
    table_name: str = None

class CTGANTrainingResult(EnsembleAttackTrainingResult):
    configs: EnsembleAttackCTGANTrainingConfig
    models: dict[Relation, CTGANModelArtifacts]
    tables: Tables
    relation_order: RelationOrder
    all_group_lengths_probabilities: GroupLengthsProbDicts


class EnsembleAttackCTGANModelRunner(EnsembleAttackModelRunner):
    def train_or_fine_tune_and_synthesize(
        self,
        dataset: pd.DataFrame,
        synthesize: bool = True,
        trained_model: EnsembleAttackTrainingResult | None = None,
    ) -> CTGANTrainingResult:
        """
        Train or fine tune a CTGAN model on the provided dataset and optionally synthesize data.

        If no trained model is provided, a new model will be trained. Otherwise, the
        provided model will be fine tuned.

        Args:
            dataset: The dataset as a pandas DataFrame.
            configs: Configuration dictionary for CTGAN.
            synthesize: Flag indicating whether to generate synthetic data after training. Defaults to True.
            trained_model: The trained model to fine tune. If None, a new model will be trained.

        Returns:
            A dataclass TrainingResult object containing:
                - save_dir: Directory where results are saved.
                - configs: Configuration dictionary used for training.
                - models: The trained models.
                - synthetic_data: The synthesized data as a pandas DataFrame, if synthesis was performed,
                otherwise, None.
        """
        assert self.training_config.metadata is not None, "Metadata is not set"
        assert self.training_config.table_name is not None, "Table name is not set"

        dataset_without_ids = dataset.drop(columns=[column_name for column_name in dataset.columns if "_id" in column_name])

        if trained_model is None:
            log(INFO, "Training new CTGAN model...")
            ctgan = CTGANSynthesizer(
                metadata=self.training_config.metadata,
                epochs=self.training_config.training.epochs,
                verbose=self.training_config.training.verbose,
            )
            model_name = "trained_ctgan_model.pkl"
        else:
            log(INFO, "Fine tuning CTGAN model...")
            ctgan = trained_model.models[(None, self.training_config.table_name)].model
            model_name = "fine_tuned_ctgan_model.pkl"

        ctgan.fit(dataset_without_ids)

        save_dir = self.training_config.general.workspace_dir / self.training_config.general.exp_name
        results_file = Path(save_dir) / model_name
        results_file.parent.mkdir(parents=True, exist_ok=True)

        ctgan.save(results_file)

        result = CTGANTrainingResult(
            save_dir=save_dir,
            configs=self.training_config,
            models={(None, self.training_config.table_name): CTGANModelArtifacts(model=ctgan, model_file_path=results_file)},
        )

        if synthesize:
            synthetic_data = ctgan.sample(num_rows=self.training_config.synthesizing.sample_size)
            result.synthetic_data = synthetic_data

        return result
