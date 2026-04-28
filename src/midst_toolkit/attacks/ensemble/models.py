"""Module containing the base classes and implementations for the Ensemble Attack model runner and training result."""

import copy
import json
import shutil
from abc import ABC, abstractmethod
from logging import INFO
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from torch.utils.data import DataLoader

from midst_toolkit.attacks.ensemble.clavaddpm_fine_tuning import clava_fine_tuning
from midst_toolkit.common.config import ClavaDDPMTrainingConfig, CTGANTrainingConfig, TrainingConfig
from midst_toolkit.common.enumerations import DataSplit
from midst_toolkit.common.logger import log
from midst_toolkit.common.variables import DEVICE
from midst_toolkit.models.clavaddpm.clustering import clava_clustering
from midst_toolkit.models.clavaddpm.data_loaders import Tables, load_tables
from midst_toolkit.models.clavaddpm.enumerations import GroupLengthsProbDicts, Relation, RelationOrder
from midst_toolkit.models.clavaddpm.synthesizer import clava_synthesizing
from midst_toolkit.models.clavaddpm.train import ClavaDDPMModelArtifacts, CTGANModelArtifacts, clava_training
from midst_toolkit.models.tabsyn.dataset import TabularDataset, preprocess
from midst_toolkit.models.tabsyn.pipeline import TabSyn
from midst_toolkit.models.tabsyn.preprocessing import get_processed_data_dir, process_data


class EnsembleAttackTrainingResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    save_dir: Path
    configs: TrainingConfig | None
    models: Any
    synthetic_data: pd.DataFrame | None = None


class EnsembleAttackModelRunner(ABC):
    training_config: TrainingConfig

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
            synthesize: Whether to synthesize data after training. The number of points to synthesize
                and the save directory is controlled by the `number_of_points_to_synthesize` and `save_dir`
                attributes of the training config. Optional, default is True.
            trained_model: The model to fine tune. If None, a new model should be trained.
                Optional, default is None.

        Returns:
            An instance of `EnsembleAttackTrainingResult` containing the training results.
        """
        raise NotImplementedError("Subclasses must implement this method.")


# ClavaDDPM implementation
class ClavaDDPMTrainingResult(EnsembleAttackTrainingResult):
    models: dict[Relation, ClavaDDPMModelArtifacts]
    tables: Tables
    relation_order: RelationOrder
    all_group_lengths_probabilities: GroupLengthsProbDicts


class EnsembleAttackClavaDDPMModelRunner(EnsembleAttackModelRunner):
    training_config: ClavaDDPMTrainingConfig

    def __init__(self, config: DictConfig):
        """
        Initialize the ensemble attack model runner with a config dictionary.

        Args:
            config: The training config from the config.yaml file for the ensemble attack model.
                Must contain the following keys:
                - shadow_training.training_json_config_paths.training_config_path:
                    The training json config path for the ClavaDDPM model.
                - shadow_training.fine_tuning_config.fine_tune_diffusion_iterations:
                    The number of diffusion iterations for the fine tuning of the ClavaDDPM model.
                - shadow_training.fine_tuning_config.fine_tune_classifier_iterations:
                    The number of classifier iterations for the fine tuning of the ClavaDDPM model.
                - shadow_training.number_of_points_to_synthesize: The number of points
                    to synthesize for the ClavaDDPM model.
        """
        with open(config.shadow_training.training_json_config_paths.training_config_path, "r") as file:
            self.training_config = ClavaDDPMTrainingConfig(**json.load(file))

        self.fine_tuning_diffusion_iterations = (
            config.shadow_training.fine_tuning_config.fine_tune_diffusion_iterations
        )
        self.fine_tuning_classifier_iterations = (
            config.shadow_training.fine_tuning_config.fine_tune_classifier_iterations
        )
        self.number_of_points_to_synthesize = config.shadow_training.number_of_points_to_synthesize

    def train_or_fine_tune_and_synthesize(
        self,
        dataset: pd.DataFrame,
        synthesize: bool = True,
        trained_model: EnsembleAttackTrainingResult | None = None,
    ) -> ClavaDDPMTrainingResult:
        """
        Train or fine tune a single-table ClavaDDPM model on the provided training set and optionally synthesize
        data using the trained/fine-tuned models.

        Args:
            dataset: The training dataset as a pandas DataFrame.
            synthesize: Flag indicating whether to generate synthetic data after training.
                The number of points to synthesize and the save directory is controlled by
                the `number_of_points_to_synthesize` and `save_dir` attributes of the
                training config. Optional, default is True.
            trained_model: The model to fine tune. If None, a new model should be trained.
                Optional, default is None.

        Returns:
            A dataclass ClavaDDPMTrainingResult object containing:
                - save_dir: Directory where results are saved.
                - configs: Configuration dictionary used for training.
                - tables: Loaded tables after clustering.
                - relation_order: Relation order of the tables.
                - all_group_lengths_probabilities: Group lengths probability dictionaries.
                - models: The trained models.
                - synthetic_data: The synthesized data as a pandas DataFrame, if synthesis was performed,
                otherwise, None.
        """
        assert self.training_config.save_dir is not None, "Save dir is not set"

        # Load tables
        tables, relation_order, _ = load_tables(self.training_config.general.data_dir, train_data={"trans": dataset})

        # Clustering on the multi-table dataset
        tables, all_group_lengths_prob_dicts = clava_clustering(
            tables,
            relation_order,
            self.training_config.save_dir,
            self.training_config.clustering,
        )

        if trained_model is None:
            # Train models
            tables, models = clava_training(
                tables,
                relation_order,
                self.training_config.save_dir,
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
                fine_tuning_diffusion_iterations=self.fine_tuning_diffusion_iterations,
                fine_tuning_classifier_iterations=self.fine_tuning_classifier_iterations,
            )

        result = ClavaDDPMTrainingResult(
            save_dir=self.training_config.save_dir,
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
            sample_scale = self.number_of_points_to_synthesize / len(tables["trans"].data)
            cleaned_tables, _, _ = clava_synthesizing(
                tables,
                relation_order,
                self.training_config.save_dir,
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
class CTGANTrainingResult(EnsembleAttackTrainingResult):
    models: dict[Relation, CTGANModelArtifacts]


class EnsembleAttackCTGANModelRunner(EnsembleAttackModelRunner):
    training_config: CTGANTrainingConfig

    def __init__(self, config: DictConfig):
        """
        Initialize the ensemble attack model runner for the CTGAN model with a config dictionary.

        Args:
            config: The training config from the config.yaml file for the ensemble attack model.
                Must contain the following keys:
                - ensemble_attack.shadow_training.training_json_config_paths.training_config_path:
                    The training json config path for the CTGAN model.
                - ensemble_attack.shadow_training.number_of_points_to_synthesize: The number of
                    points to synthesize for the CTGAN model.
                - ensemble_attack.table_name: The name of the table the CTGAN model is being trained on.
                - ensemble_attack.shadow_training.model_config.training.epochs: The number of epochs
                    to train the CTGAN shadow model.
                - ensemble_attack.shadow_training.model_config.training.verbose: Whether to print
                    verbose output during training of the CTGAN shadow model.
        """
        with open(config.ensemble_attack.shadow_training.training_json_config_paths.training_config_path, "r") as file:
            self.training_config = CTGANTrainingConfig(**json.load(file))
        self.number_of_points_to_synthesize = config.ensemble_attack.shadow_training.number_of_points_to_synthesize
        self.table_name = config.ensemble_attack.table_name
        self.training_epochs = config.ensemble_attack.shadow_training.model_config.training.epochs
        self.training_verbose = config.ensemble_attack.shadow_training.model_config.training.verbose

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
            synthesize: Flag indicating whether to generate synthetic data after training.
                The number of points to synthesize and the save directory is controlled by
                the `number_of_points_to_synthesize` and `save_dir` attributes of the training
                config. Optional, default is True.
            trained_model: The trained model to fine tune. If None, a new model will be trained.

        Returns:
            A dataclass TrainingResult object containing:
                - save_dir: Directory where results are saved.
                - configs: Configuration dictionary used for training.
                - models: The trained models.
                - synthetic_data: The synthesized data as a pandas DataFrame, if synthesis was performed,
                otherwise, None.
        """
        assert self.training_config.save_dir is not None, "Save dir is not set"
        assert self.table_name is not None, "Table name is not set"

        domain_file_path = Path(self.training_config.general.data_dir) / f"{self.table_name}_domain.json"
        with open(domain_file_path, "r") as file:
            domain_dictionary = json.load(file)

        metadata, dataset_without_ids = get_single_table_svd_metadata(dataset, domain_dictionary)

        if trained_model is None:
            log(INFO, "Training new CTGAN model...")
            ctgan = CTGANSynthesizer(
                metadata=metadata,
                epochs=self.training_epochs,
                verbose=self.training_verbose,
            )
            model_name = "trained_ctgan_model.pkl"
        else:
            log(INFO, "Fine tuning CTGAN model...")
            ctgan = trained_model.models[(None, self.table_name)].model
            model_name = "fine_tuned_ctgan_model.pkl"

        ctgan.fit(dataset_without_ids)

        results_file = self.training_config.save_dir / model_name
        results_file.parent.mkdir(parents=True, exist_ok=True)

        ctgan.save(results_file)

        result = CTGANTrainingResult(
            save_dir=self.training_config.save_dir,
            configs=self.training_config,
            models={(None, self.table_name): CTGANModelArtifacts(model=ctgan, model_file_path=results_file)},
        )

        if synthesize:
            synthetic_data = ctgan.sample(num_rows=self.number_of_points_to_synthesize)
            result.synthetic_data = synthetic_data

        return result


def get_single_table_svd_metadata(
    data: pd.DataFrame,
    domain_dictionary: dict[str, Any] | None = None,
) -> tuple[SingleTableMetadata, pd.DataFrame]:
    """
    Get the metadata for a single-table dataset for SDV models.

    Args:
        data: The dataframe containing the data.
        domain_dictionary: The domain dictionary containing metadata about the data columns.

    Returns:
        A tuple containing the metadata and the dataframe without the id columns.
    """
    metadata = SingleTableMetadata()
    data_without_ids = data.drop(columns=[column_name for column_name in data.columns if "_id" in column_name])
    metadata.detect_from_dataframe(data_without_ids)  # Starts up the metadata info from the dataframe's columns.

    if domain_dictionary is not None:
        for column_name in data_without_ids.columns:
            if domain_dictionary[column_name]["type"] == "discrete":
                if domain_dictionary[column_name]["size"] < 1000:
                    metadata.update_column(
                        column_name=column_name,
                        sdtype="categorical",
                    )
                else:
                    metadata.update_column(
                        column_name=column_name,
                        sdtype="numerical",
                    )
            else:
                metadata.update_column(
                    column_name=column_name,
                    sdtype="numerical",
                )

    metadata.remove_primary_key()

    return metadata, data_without_ids


class TabSynTrainingResult(EnsembleAttackTrainingResult):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tabsyn_config: dict[str, Any]


class EnsembleAttackTabSynModelRunner(EnsembleAttackModelRunner):
    tabsyn_config: dict[str, Any]

    def __init__(self, config: DictConfig):
        """
        Initialize the ensemble attack model runner for the TabSyn model with a config dictionary.

        Args:
            config: The config from the config.yaml file for the ensemble attack model.
                Must contain the following keys:
                - tabsyn_config: The tabsyn config path for the TabSyn model.
                - data_dir: The data directory for the TabSyn model.
                - results_dir: The results directory for the TabSyn model.
                - table_name: The name of the table the TabSyn model is being trained on.
        """
        self.tabsyn_config = config.tabsyn_config
        self.data_dir = Path(config.data_dir)
        self.results_dir = Path(config.results_dir)
        self.table_name = config.table_name
        self.model_save_dir = self.results_dir / self.table_name
        self.vae_save_dir = self.model_save_dir / "vae"

    def train_or_fine_tune_and_synthesize(
        self,
        dataset: pd.DataFrame,
        synthesize: bool = True,
        trained_model: EnsembleAttackTrainingResult | None = None,
    ) -> TabSynTrainingResult:
        """
        Train or fine tune a TabSyn model on the provided dataset and optionally synthesize data.

        Args:
            dataset: The dataset as a pandas DataFrame.
            synthesize: Flag indicating whether to generate synthetic data after training.
                The number of points to synthesize and the save directory is controlled by
                the `number_of_points_to_synthesize` and `save_dir` attributes of the training
                config. Optional, default is True.
            trained_model: The trained model to fine tune. If None, a new model will be trained.
        """
        log(INFO, "Training TabSyn model...")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        tabsyn: TabSyn
        synthetic_data: pd.DataFrame | None = None
        if trained_model is None:
            tabsyn = self._train()
        else:
            log(INFO, "Instantiating the TabSyn model...")
            tabsyn = copy.deepcopy(trained_model.models)

        if synthesize:
            synthetic_data = self._synthesize(tabsyn)

        return TabSynTrainingResult(
            save_dir=self.results_dir,
            configs=None,
            tabsyn_config=self.tabsyn_config,
            models=tabsyn,
            synthetic_data=synthetic_data,
        )

    def _dataset_and_ref_dataset_paths(self) -> tuple[Path, Path]:
        # The preprocess function below expects 2 folders of preprocessed data:
        # 1. the dataset to be processed, which can be a subset of the full dataset
        # 2. the full dataset, which is used to unsure it will get all the categories for the categorical features
        # Here we are making the dataset #2 (full dataset) by copying the dataset #1
        dataset_path = get_processed_data_dir(self.data_dir) / self.table_name
        ref_dataset_path = Path(f"{dataset_path}_all")
        shutil.rmtree(ref_dataset_path, ignore_errors=True)
        shutil.copytree(dataset_path, ref_dataset_path)

        return dataset_path, ref_dataset_path

    def _train(self) -> TabSyn:
        log(INFO, "Training new TabSyn model...")

        process_data(self.table_name, self.data_dir, self.data_dir)

        log(INFO, "Preprocessing data...")

        dataset_path, ref_dataset_path = self._dataset_and_ref_dataset_paths()

        # preprocess the data
        # TODO: refactor the return of the preprocess function so we don't need to ignore mypy here
        numerical_features, categorical_features, categories, d_numerical = preprocess(  # type: ignore[misc]
            dataset_path=dataset_path,
            ref_dataset_path=ref_dataset_path,
            transforms=self.tabsyn_config["transforms"],
            task_type=self.tabsyn_config["task_type"],
        )

        # separate train and test data
        numerical_features_train = numerical_features[DataSplit.TRAIN.value]
        numerical_features_test = numerical_features[DataSplit.TEST.value]
        categorical_features_train = categorical_features[DataSplit.TRAIN.value]
        categorical_features_test = categorical_features[DataSplit.TEST.value]

        # convert to float tensor
        numerical_features_train = torch.tensor(numerical_features_train).float()
        numerical_features_test = torch.tensor(numerical_features_test).float()
        categorical_features_train = torch.tensor(categorical_features_train)
        categorical_features_test = torch.tensor(categorical_features_test)

        log(INFO, "Loading the dataset...")

        # create dataset module
        train_data = TabularDataset(numerical_features_train.float(), categorical_features_train)

        # move test data to gpu if available
        numerical_features_test = numerical_features_test.float().to(DEVICE)
        categorical_features_test = categorical_features_test.to(DEVICE)

        # create train dataloader
        train_loader: DataLoader[TabularDataset] = DataLoader[TabularDataset](
            # Ignoring here because this is expecting the dataset to be subclass of torch's Dataset but it isn't
            train_data,  # type: ignore[arg-type]
            batch_size=self.tabsyn_config["train"]["vae"]["batch_size"],
            shuffle=True,
            num_workers=self.tabsyn_config["train"]["vae"]["num_dataset_workers"],
        )

        log(INFO, "Instantiating the TabSyn model...")

        # Instantiate the model
        tabsyn = TabSyn(
            train_loader,
            numerical_features_test,
            categorical_features_test,
            num_numerical_features=d_numerical,
            num_classes=categories,
            device=DEVICE,
        )
        self.vae_save_dir.mkdir(parents=True, exist_ok=True)

        ###### A. Train the VAE model ######

        log(INFO, "Training the TabSyn VAE model...")

        # instantiate VAE model for training
        tabsyn.instantiate_vae(
            **self.tabsyn_config["model_params"],
            optim_params=self.tabsyn_config["train"]["optim"]["vae"],
        )

        tabsyn.train_vae(
            **self.tabsyn_config["loss_params"],
            num_epochs=self.tabsyn_config["train"]["vae"]["num_epochs"],
            save_path=self.vae_save_dir,
        )

        # embed all inputs in the latent space
        tabsyn.save_vae_embeddings(
            numerical_features_train,
            categorical_features_train,
            vae_ckpt_dir=self.vae_save_dir,
        )
        tabsyn.save_embeddings_attributes(vae_ckpt_dir=self.vae_save_dir)

        ###### B. Train the Diffusion model ######

        log(INFO, "Training the TabSyn Diffusion model...")

        # load latent space embeddings
        train_z, _ = tabsyn.load_latent_embeddings(self.vae_save_dir)  # train_z dim: B x in_dim

        # normalize embeddings
        latent_train_data = (train_z - train_z.mean(0)) / 2

        # create data loader
        latent_train_loader: DataLoader[torch.Tensor] = DataLoader[torch.Tensor](
            # Ignoring the type checker here because our code in tabsyn.train_diffusion
            # works with plain Tensor and not with TensorDataset
            latent_train_data,  # type: ignore[arg-type]
            batch_size=self.tabsyn_config["train"]["diffusion"]["batch_size"],
            shuffle=True,
            num_workers=self.tabsyn_config["train"]["diffusion"]["num_dataset_workers"],
        )

        # instantiate diffusion model for training
        tabsyn.instantiate_diffusion(
            in_dim=train_z.shape[1],
            hid_dim=train_z.shape[1],
            optim_params=self.tabsyn_config["train"]["optim"]["diffusion"],
        )

        # train diffusion model
        tabsyn.train_diffusion(
            latent_train_loader,
            num_epochs=self.tabsyn_config["train"]["diffusion"]["num_epochs"],
            ckpt_path=self.model_save_dir,
        )

        log(INFO, "Training Done!")

        return tabsyn

    def _synthesize(self, tabsyn: TabSyn) -> pd.DataFrame:
        ###### Load the model ######

        # instantiate VAE model
        tabsyn.instantiate_vae(**self.tabsyn_config["model_params"], optim_params=None)

        # load latent embeddings attributes of input data
        train_z_att = tabsyn.load_embeddings_attributes(self.vae_save_dir)
        token_dim = train_z_att["token_dim"]
        in_dim = train_z_att["in_dim"]
        hid_dim = train_z_att["hid_dim"]

        # instantiate diffusion model
        tabsyn.instantiate_diffusion(in_dim=in_dim, hid_dim=hid_dim, optim_params=None)

        # load state from checkpoint
        tabsyn.load_model_state(ckpt_dir=self.model_save_dir, dif_ckpt_name="model.pt")

        ###### Synthesize data ######

        dataset_path, ref_dataset_path = self._dataset_and_ref_dataset_paths()

        # get inverse tokenizers
        # TODO: refactor the return of the preprocess function so we don't need to ignore mypy here
        _, _, _, _, num_inverse, cat_inverse = preprocess(  # type: ignore[misc]
            dataset_path=dataset_path,
            ref_dataset_path=ref_dataset_path,
            transforms=self.tabsyn_config["transforms"],
            task_type=self.tabsyn_config["task_type"],
            inverse=True,
        )

        synthetic_data_dir = self.results_dir / self.table_name / "synthetic_data"
        synthetic_data_dir.mkdir(parents=True, exist_ok=True)

        # load data info file
        with open(dataset_path / "info.json", "r") as file:
            data_info = json.load(file)

        data_info["token_dim"] = token_dim

        # sample data
        num_samples = train_z_att["num_samples"]
        in_dim = train_z_att["in_dim"]
        mean_input_emb = train_z_att["mean_input_emb"]
        tabsyn.sample(
            num_samples,
            in_dim,
            mean_input_emb,
            info=data_info,
            num_inverse=num_inverse,
            cat_inverse=cat_inverse,
            save_path=synthetic_data_dir / f"{self.table_name}_synthetic.csv",
        )

        log(INFO, "Synthesizing Done!")

        return pd.read_csv(synthetic_data_dir / f"{self.table_name}_synthetic.csv")
