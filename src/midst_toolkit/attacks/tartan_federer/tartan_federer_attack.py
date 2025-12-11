from __future__ import annotations

import csv
import os
from collections.abc import Generator
from logging import INFO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import Tensor, nn
from tqdm import tqdm

from midst_toolkit.attacks.tartan_federer.classification import MLP, fit_model
from midst_toolkit.attacks.tartan_federer.data_utils import (
    CustomUnpickler,
    evaluate_attack_performance,
    load_multi_table_customized,
    prepare_population_dataset_for_attack,
)
from midst_toolkit.common.enumerations import DataSplit
from midst_toolkit.common.logger import configure, log
from midst_toolkit.common.variables import DEVICE
from midst_toolkit.models.clavaddpm.data_loaders import prepare_fast_dataloader
from midst_toolkit.models.clavaddpm.dataset import (
    Dataset,
    TableMetadata,
    Transformations,
    get_categorical_and_numerical_column_names,
    transform_dataset,
)
from midst_toolkit.models.clavaddpm.dataset_transformations import TargetInfo
from midst_toolkit.models.clavaddpm.enumerations import IsTargetConditioned, TargetType
from midst_toolkit.models.clavaddpm.gaussian_multinomial_diffusion import (
    GaussianMultinomialDiffusion,
)


# TODO: remove additional_timestep from here and other places since not needed for single table attack
def mixed_loss(
    diffusion_model: GaussianMultinomialDiffusion,
    features: Tensor,
    outputs: dict[str, Tensor],
    noise: list[list[float]],
    parallel_batch: int,
    additional_timestep: int,
    timestep: int,
) -> Tensor:
    """
    Compute the loss function for the Tartan Federer classifier.

    Args:
        diffusion_model: The diffusion model on which to measure the loss
        features: The tabular features to measure the loss on.
        outputs: Outputs from the model
        noise: List of noises for the model process.
        parallel_batch: Number of parallel batches for processing.
        additional_timestep: Additional value to be passed to the loss function.
        timestep: Value of the time step `t` to be used in the loss computation.

    Returns:
        Compute the loss values to help train or predict with the Tartan Federer Classifier.
    """
    device = features.device
    batch_size = features.shape[0]
    numerical_features = features[:, : diffusion_model.num_numerical_features]
    categorical_features = features[:, diffusion_model.num_numerical_features :]

    noise_tensor = torch.tensor(noise, device=device, dtype=torch.float)
    # Here we're repeating the noise tensor for each sample in the dataset so that each point gets the same set of
    # different noise values. This happens because parallel_batch is set to num_noise_per_time_step in preceding
    # calling functions
    batch_noise = noise_tensor.repeat(batch_size, 1)

    # TODO: Handle the categorical features more effectively. Because the numerical features were originally ignored
    # in the diffusion model and thus are ignored in this attack construction.
    numerical_features = numerical_features.repeat_interleave(parallel_batch, dim=0)
    categorical_features = categorical_features.repeat_interleave(parallel_batch, dim=0)

    # Note that the shape here is not equivalent to batch_size after the interleave
    zero_timestep = torch.zeros(numerical_features.shape[0], device=DEVICE).long()
    current_timestep = zero_timestep + timestep

    # forward x_num_t with (t + additional_t) timesteps
    # TODO: Expand this to also include categorical features
    numerical_features_t = diffusion_model.gaussian_q_sample(
        numerical_features, current_timestep + additional_timestep, noise=batch_noise
    )

    # predict noises with t timesteps
    predicted_noise = diffusion_model._denoise_fn(numerical_features_t, current_timestep, **outputs)
    current_loss = diffusion_model._gaussian_loss(
        predicted_noise, batch_noise, batch_noise, current_timestep, batch_noise
    )
    return current_loss.reshape(-1, parallel_batch)


def make_dataset_from_df_with_loaded(
    data: pd.DataFrame,
    transformation: Transformations,
    is_target_conditioned: IsTargetConditioned,
    table_metadata: TableMetadata,
    label_encoders: dict[int, LabelEncoder],
    numerical_transform: StandardScaler | None = None,
    noise_scale: float = 0,
) -> Dataset:
    """
    Create a dataset using artifacts drawn from a checkpoint.

    Args:
        data: Raw data to be used for the checkpoint.
        transformation: Transformations that one might apply to the dataset, including NaN policies etc.
        is_target_conditioned: Enum indicating how, if at all, the model uses a target for generation conditioning.
        table_metadata: Meta data about the table or tables.
        label_encoders: Encoders that were used to encode the categorical data.
        numerical_transform: Transformations that should be applied to the numerical data. Defaults to None.
        noise_scale: he scale of the noise to add to the categorical features. Noise is drawn from a normal
            distribution with standard deviation of ``noise_scale``. Defaults to 0.

    Returns:
        A full dataset constructed of the various pieces.
    """
    categorical_column_names, numerical_column_names = get_categorical_and_numerical_column_names(
        table_metadata,
        is_target_conditioned,
    )

    numerical_features = {"train": data[numerical_column_names].values.astype(np.float32)}
    categorical_features = {"train": data[categorical_column_names].to_numpy(dtype=np.str_)}
    targets = {"train": data[[table_metadata.target_column_name]].values.astype(np.float32)}

    if len(categorical_column_names) > 0:
        all_categorical_features = categorical_features["train"]
        encoded_categorical_features = []
        for column_index in range(all_categorical_features.shape[1]):
            encoded_column = (
                label_encoders[column_index].transform(all_categorical_features[:, column_index]).astype(float)
            )
            if noise_scale > 0:
                # add noise
                encoded_column += np.random.normal(0, noise_scale, encoded_column.shape)
            encoded_categorical_features.append(encoded_column)

        categorical_features["train"] = np.vstack(encoded_categorical_features).T

    if len(numerical_column_names) >= 0:
        numerical_features["train"] = np.concatenate(
            (numerical_features["train"], categorical_features["train"]), axis=1
        )
    else:
        numerical_features = categorical_features

    target_info = TargetInfo(policy=None, mean=None, std=None)
    dataset = Dataset(
        numerical_features=numerical_features,
        categorical_features=None,
        target=targets,
        target_info=target_info,
        task_type=table_metadata.task_type,
        n_classes=table_metadata.n_classes,
        categorical_transform=None,
        numerical_transform=numerical_transform,
    )
    return transform_dataset(dataset, transformation, None)


def get_dataset(
    data_path: Path,
    target_model_dir: Path,
    train_name: str = "train_with_id.csv",
    batch_size: int = 32,
    meta_dir: Path = Path(""),
) -> list[tuple[Generator[list[Tensor]], int, Dataset]]:
    """
    Creates a data loader and dataset from loaded artifacts.

    Args:
        data_path: The directory to load the dataset and information from.
        target_model_dir: The directory to load the model artifact from. Should contain a file with a name formatted as
            {parent}_{child}_ckpt.pkl Defaults to None.
        train_name: Name of the file containing the table data. This should exist in the ``data_dir`` path.
            Defaults to "train_with_id.csv".
        batch_size: Size of the batches for the data loader. Defaults to 32.
        meta_dir: A separate path containing the meta data information about the tables and datasets.
            If None, this function looks for 'dataset_meta.json' in the ``data_dir`` path. Defaults to Path("").
        meta_dir: An optional separate path containing the meta data information about the tables and datasets.
            If None, this function looks for 'dataset_meta.json' in the ``data_dir`` path. Defaults to None.
        train_name: Name of the file containing the table data. This should exist in the ``data_dir`` path.
            Defaults to "train.csv".

    Raises:
        NotImplementedError: If we're trying to load a multi-table dataset, as the attack isn't implemented for
            multi-table yet.

    Returns:
        A tuple with a dataloader, size of the dataset, and dataset object.
    """
    tables, relation_order, _ = load_multi_table_customized(data_path, meta_dir=meta_dir, train_name=train_name)

    train_loader_list = []
    if len(relation_order) == 1:
        parent, child = relation_order[0]

        df_with_id = tables[child].data
        df_without_id = df_with_id.drop(columns=[col for col in df_with_id.columns if "_id" in col])

        file_path = target_model_dir / f"{parent}_{child}_ckpt.pkl"

        with open(file_path, "rb") as f:
            model = CustomUnpickler(f).load()

        df_without_id["placeholder"] = df_without_id.index

        transformations = model.transformations
        numerical_transform = model.dataset.numerical_transform

        dataset = make_dataset_from_df_with_loaded(
            df_without_id,
            transformations,
            is_target_conditioned=model.model_parameters.is_target_conditioned,
            table_metadata=model.table_metadata,
            noise_scale=0,
            numerical_transform=numerical_transform,
            label_encoders=model.label_encoders,
        )
        if dataset.numerical_features is not None:
            dataset.numerical_features["test"] = dataset.numerical_features["train"]
        if dataset.categorical_features is not None:
            dataset.categorical_features["test"] = dataset.categorical_features["train"]
        dataset.target["test"] = dataset.target["train"]

        train_loader = prepare_fast_dataloader(
            dataset,
            split=DataSplit.TEST,
            batch_size=batch_size,
            target_type=TargetType.LONG,
        )
        assert dataset.numerical_features is not None, "Numerical features are assumed to be present for this attack"
        train_loader_list.append((train_loader, dataset.numerical_features["test"].shape[0], dataset))

        return train_loader_list

    raise NotImplementedError("Multitable with more than one relation not supported yet")


def get_score(
    data_path: Path,
    save_dir: Path,
    input_noise: list[list[float]],
    model_type: str,
    meta_dir: Path,
    challenge_name: str,
    batch_size: int,
    parallel_batch: int,
    additional_timestep: int,
    timestep: int,
) -> torch.Tensor:
    """
    Computes the score for a given dataset using a diffusion model.

    Args:
        data_path: Path to the dataset.
        save_dir: Directory where model checkpoints are saved.
        input_noise: List of noise values to be used in the loss computation.
        model_type: Type of model to use (e.g., "tabddpm").
        meta_dir: Path to the metadata directory.
        challenge_name: Name of the challenge dataset.
        batch_size: Batch size for data loading.
        parallel_batch: Number of parallel batches for processing.
        additional_timestep: Additional value to be passed to the loss function.
        timestep: Value of the time step `t` to be used in the loss computation.

    Returns:
        A tensor containing the computed loss values.

    Raises:
        ValueError: If the specified `model_type` is not supported.
        AssertionError: If required model checkpoint files are not found or if `iter_max` is not equal to 1.
    """
    if model_type == "tabddpm":
        relation_order = [("None", "trans")]
    elif model_type == "tabsyn":
        raise ValueError("Haven't done it yet!")

    train_loader_list = get_dataset(
        data_path,
        save_dir,
        train_name=challenge_name,
        batch_size=batch_size,
        meta_dir=meta_dir,
    )

    assert len(relation_order) == 1, "Attack not yet implemented for multi-table setting"
    for parent, child in relation_order:
        filepath = save_dir / f"{parent}_{child}_ckpt.pkl"
        assert os.path.exists(filepath)

        # get the diffusion model
        with open(filepath, "rb") as f:
            model = CustomUnpickler(f).load()

        diffusion_model = model.diffusion.to(DEVICE)
        assert isinstance(diffusion_model, GaussianMultinomialDiffusion)

        train_loader, dataset_size, _ = train_loader_list[0]
        assert dataset_size // batch_size, (
            f"Batch size ({batch_size}) must be less than or equal to the dataset size ({dataset_size})"
        )

        features, labels = next(train_loader)
        outputs = {"y": labels.long().to(DEVICE)}
        features = features.to(DEVICE)

        with torch.no_grad():
            # get loss here
            loss = mixed_loss(
                diffusion_model=diffusion_model,
                features=features,
                outputs=outputs,
                noise=input_noise,
                parallel_batch=parallel_batch,
                additional_timestep=additional_timestep,
                timestep=timestep,
            )

    # TODO: Should we be summing this loss or something? We're only going to get the last loss in the iteration.
    return loss


def filter_dataframe(
    dataframe_to_filter: pd.DataFrame, dataframe_to_filter_by: pd.DataFrame, columns_for_filtration: list[str]
) -> pd.DataFrame:
    """
    Filters a dataframe by the indices of another dataframe. This will return only those rows in
    ``dataframe_to_filter`` that are NOT present in ``dataframe_to_filter_by`` when considering the column names in
    ``columns_for_filtration``.

    Args:
        dataframe_to_filter: Dataframe to filter
        dataframe_to_filter_by: Dataframe to filter by
        columns_for_filtration: Column names that we will use to compare rows in ``dataframe_to_filter`` and
            ``dataframe_to_filter_by``

    Returns:
        Filtered dataframe.
    """
    indices_to_keep = ~dataframe_to_filter.set_index(columns_for_filtration).index.isin(
        dataframe_to_filter_by.set_index(columns_for_filtration).index
    )
    return dataframe_to_filter[indices_to_keep]


def prepare_dataframe(
    model_dir: Path,
    merged_data: pd.DataFrame,
    columns_for_deduplication: list[str],
    samples_per_model: int,
    mia_dataset_name: str,
) -> pd.DataFrame:
    """
    Prepare the dataframes for Tartan Federer Attack Classifier training.

    Args:
        model_dir: Model directory from which to load data. This directory must contain a file named
            "train_with_id.csv" and "data_for_training_MIA.csv"
        merged_data: Dataframe constructed with the ``prepare_population_dataset_for_attack`` function.
        columns_for_deduplication: Columns to use in filtering the dataframes.
        samples_per_model: Number of samples to draw from the prepared data for model training.
        mia_dataset_name: Name of the MIA dataset file to be saved.

    Returns:
        Filtered dataframe reading for classifier training (or testing)
    """
    raw_data = pd.read_csv(model_dir / "train_with_id.csv")

    df_exclusive = filter_dataframe(merged_data, raw_data, columns_for_deduplication)

    data_exclusive = df_exclusive.sample(samples_per_model)
    data_from_train = raw_data.sample(samples_per_model)

    df_data = pd.concat([data_exclusive, data_from_train], ignore_index=True)
    df_data.to_csv(model_dir / mia_dataset_name, index=False)

    return filter_dataframe(merged_data, df_data, columns_for_deduplication)


def train_tartan_federer_attack_classifier(
    train_indices: list[int],
    val_indices: list[int],
    timesteps: list[int],
    columns_for_deduplication: list[str],
    additional_timesteps: list[int],
    num_noise_per_time_step: int,
    samples_per_train_model: int,
    sample_per_val_model: int,
    classifier_num_epochs: int,
    classifier_hidden_dim: int,
    classifier_learning_rate: float,
    model_type: str,
    model_data_dir: Path,
    results_path: Path,
    target_model_subdir: Path,
    meta_dir: Path,
) -> tuple[list[list[float]], nn.Module]:
    """
    Train a Tartan Federer MIA classifier using the provided information.

    Args:
        train_indices: List of model indices used to extract features for training the binary classifier.
        val_indices: List of model indices used to extract features for validating the binary classifier.
        timesteps: List of timesteps of the diffusion model to be used in the loss computation.
        columns_for_deduplication: List of column names used to ensure that training, validation, and test datasets
                                   are distinct. For example, this list might include ["trans_id", "balance"] for the
                                   Berka dataset in the MIDST competition.
        additional_timesteps: List of additional timesteps to be used in the loss computation.
        num_noise_per_time_step: Number of Gaussian noise samples to be used for each timestep in the loss computation.
        samples_per_train_model: Number of samples drawn from the training data (members) of train indices and
                                 non-members for training the binary classifier.
        sample_per_val_model: Number of samples drawn from the training data (members) of validation indices and
                              non-members for validating the binary classifier.
        classifier_num_epochs: Number of epochs used to train the MLP as the binary classifier.
        classifier_hidden_dim: The width of the 3-layer MLP trained as the binary classifier.
        classifier_learning_rate: Learning rate used to train the binary classifier.
        model_type: Type of diffusion model, e.g., "tabddpm" for ClavaDDPM-single-table.
        model_data_dir: Base directory containing all the trained diffusion models.
        results_path: Directory where the attack results will be saved.
        target_model_subdir: Sub-directory within each model directory containing the trained diffusion model
                             checkpoint.
        meta_dir: Directory containing metadata about the datasets, including a file named `dataset_meta.json`.

    Returns:
        A tuple containing the noise samples used in the loss computation and the trained classifier model.
    """
    population_df_for_training = prepare_population_dataset_for_attack(
        model_indices=train_indices,
        model_type=model_type,
        models_base_dir=model_data_dir,
        columns_for_deduplication=columns_for_deduplication,
    )

    population_df_for_validation = prepare_population_dataset_for_attack(
        model_indices=val_indices,
        model_type=model_type,
        models_base_dir=model_data_dir,
        columns_for_deduplication=columns_for_deduplication,
    )

    noise_dimension = len([col for col in population_df_for_training.columns if "_id" not in col])
    input_noise = [np.random.normal(size=noise_dimension).tolist() for _ in range(num_noise_per_time_step)]
    input_dimension = len(input_noise) * len(timesteps) * len(additional_timesteps)

    total_data_num_for_train = samples_per_train_model * 2 * len(train_indices)
    x_train = np.zeros([total_data_num_for_train, input_dimension])
    y_train = np.zeros([total_data_num_for_train])

    if val_indices is not None:
        total_data_num_for_validation = sample_per_val_model * 2 * len(val_indices)
        x_val = np.zeros([total_data_num_for_validation, input_dimension])
        y_val = np.zeros([total_data_num_for_validation])
    else:
        x_val, y_val = None, None

    regression_model = MLP(input_dim=input_dimension, hidden_dim=classifier_hidden_dim).to(DEVICE)

    train_count = 0
    val_count = 0
    val_indices = [] if val_indices is None else val_indices

    model_folders_indices = np.concatenate((train_indices, val_indices))
    for model_number in tqdm(model_folders_indices, desc="Processing models", unit="model"):
        model_folder = f"{model_type}_{model_number}"
        model_dir = model_data_dir / model_folder
        model_path = model_dir / target_model_subdir

        if model_number in train_indices:
            population_df_for_training = prepare_dataframe(
                model_dir,
                population_df_for_training,
                columns_for_deduplication,
                samples_per_train_model,
                "data_for_training_MIA.csv",
            )

        elif model_number in val_indices:
            population_df_for_validation = prepare_dataframe(
                model_dir,
                population_df_for_validation,
                columns_for_deduplication,
                sample_per_val_model,
                "data_for_validating_MIA.csv",
            )

        timestep_count = 0
        for timestep in timesteps:
            for additional_timestep in additional_timesteps:
                if model_number in train_indices:
                    batch_size = samples_per_train_model * 2

                    predictions = get_score(
                        model_dir,
                        model_path,
                        input_noise,
                        model_type,
                        meta_dir=meta_dir,
                        challenge_name="data_for_training_MIA.csv",
                        batch_size=batch_size,
                        parallel_batch=num_noise_per_time_step,
                        additional_timestep=additional_timestep,
                        timestep=timestep,
                    )

                    x_train[
                        samples_per_train_model * 2 * train_count : samples_per_train_model * 2 * (train_count + 1),
                        timestep_count * num_noise_per_time_step : (timestep_count + 1) * num_noise_per_time_step,
                    ] = predictions.detach().squeeze().cpu().numpy()

                    y_train[
                        samples_per_train_model * 2 * train_count : samples_per_train_model * 2 * (train_count + 1)
                    ] = np.concatenate(
                        [
                            np.zeros(samples_per_train_model),
                            np.ones(samples_per_train_model),
                        ]
                    )

                    timestep_count += 1

                elif model_number in val_indices:
                    batch_size = sample_per_val_model * 2
                    predictions = get_score(
                        model_dir,
                        model_path,
                        input_noise,
                        model_type,
                        meta_dir=meta_dir,
                        challenge_name="data_for_validating_MIA.csv",
                        batch_size=batch_size,
                        parallel_batch=num_noise_per_time_step,
                        additional_timestep=additional_timestep,
                        timestep=timestep,
                    )
                    assert x_val is not None and y_val is not None
                    x_val[
                        sample_per_val_model * 2 * val_count : sample_per_val_model * 2 * (val_count + 1),
                        timestep_count * num_noise_per_time_step : (timestep_count + 1) * num_noise_per_time_step,
                    ] = predictions.detach().squeeze().cpu().numpy()

                    y_val[sample_per_val_model * 2 * val_count : sample_per_val_model * 2 * (val_count + 1)] = (
                        np.concatenate([np.zeros(sample_per_val_model), np.ones(sample_per_val_model)])
                    )

                    timestep_count += 1

        if model_number in train_indices:
            train_count += 1
        elif model_number in val_indices:
            val_count += 1

    fitted_regression_model = fit_model(
        regression_model=regression_model,
        train_features=x_train,
        train_targets=y_train,
        validation_features=x_val,
        validation_targets=y_val,
        num_epochs=classifier_num_epochs,
        best_model_checkpoint_dir=results_path,
        learning_rate=classifier_learning_rate,
    )
    return input_noise, fitted_regression_model


def tartan_federer_attack(
    train_indices: list[int],
    val_indices: list[int] | None,
    test_indices: list[int],
    columns_for_deduplication: list[str],
    timesteps: list[int],
    additional_timesteps: list[int],
    num_noise_per_time_step: int,
    samples_per_train_model: int,
    sample_per_val_model: int,
    classifier_num_epochs: int,
    classifier_hidden_dim: int,
    classifier_learning_rate: float,
    model_type: str,
    predictions_file_format: str,
    model_data_dir: Path,
    meta_dir: Path,
    target_model_subdir: Path,
    results_path: Path,
    save_results: bool = True,
) -> tuple[Any, Any, Any]:
    """
    Executes the Tartan Federer Membership Inference Attack (MIA) on a set of diffusion models.

    Args:
        train_indices: List of model indices used to extract features for training the binary classifier.
        val_indices: List of model indices used to extract features for validating the binary classifier.
                     If None, no validation is performed.
        test_indices: List of model indices to report the final MIA performance on their respective challenge points.
        columns_for_deduplication: List of column names used to ensure that training, validation, and test datasets
                                   are distinct. For example, this list might include ["trans_id", "balance"] for the
                                   Berka dataset in the MIDST competition.
        timesteps: List of timesteps of the diffusion model to be used in the loss computation.
        additional_timesteps: List of additional timesteps to be used in the loss computation for the multi-table
                              attack. Defaults to [0] for single-table attacks.
        num_noise_per_time_step: Number of Gaussian noise samples to be used for each timestep in the loss computation.
        samples_per_train_model: Number of samples drawn from the training data (members) of train indices and
                                 non-members for training the binary classifier.
        sample_per_val_model: Number of samples drawn from the training data (members) of validation indices and
                              non-members for validating the binary classifier.
        classifier_num_epochs: Number of epochs used to train the MLP as the binary classifier.
        classifier_hidden_dim: The width of the 3-layer MLP trained as the binary classifier.
        classifier_learning_rate: Learning rate used to train the binary classifier.
        model_type: Type of diffusion model, e.g., "tabddpm" for ClavaDDPM-single-table.
        predictions_file_format: Format for naming the MIA prediction files.
        model_data_dir: Base directory containing all the trained diffusion models.
        meta_dir: Directory containing metadata about the datasets, including a file named `dataset_meta.json`.
        target_model_subdir: Sub-directory within each model directory containing the trained diffusion model
                             checkpoint.
        results_path: Directory where the training log, attack results, and the binary classifier will be saved.
        save_results: Boolean flag indicating whether to save the results to a text file. Defaults to True.

    Returns:
        A tuple containing the MIA performance metrics for the training, validation, and test datasets.
    """
    configure(identifier="tartan_federer_attack", filename=str(results_path / "tartan_federer_attack.log"))
    log(INFO, "Starting Tartan Federer Attack.")

    os.makedirs(results_path, exist_ok=True)
    val_indices = [] if val_indices is None else val_indices

    input_noise, regression_model = train_tartan_federer_attack_classifier(
        train_indices=train_indices,
        val_indices=val_indices,
        columns_for_deduplication=columns_for_deduplication,
        samples_per_train_model=samples_per_train_model,
        sample_per_val_model=sample_per_val_model,
        model_type=model_type,
        model_data_dir=model_data_dir,
        target_model_subdir=target_model_subdir,
        classifier_hidden_dim=classifier_hidden_dim,
        num_noise_per_time_step=num_noise_per_time_step,
        timesteps=timesteps,
        additional_timesteps=additional_timesteps,
        classifier_num_epochs=classifier_num_epochs,
        results_path=results_path,
        meta_dir=meta_dir,
        classifier_learning_rate=classifier_learning_rate,
    )

    predictions_file_name = f"{predictions_file_format}.csv"

    model_folders_indices = np.concatenate((train_indices, val_indices, test_indices))
    for model_number in tqdm(model_folders_indices, desc="Processing models", unit="model"):
        model_folder: str = f"{model_type}_{model_number}"
        model_dir = model_data_dir / model_folder
        model_path = model_dir / target_model_subdir

        batch_size = 200
        batches_of_predictions = []
        for timestep in timesteps:
            # TODO Make this not [0] for multi-table?
            for additional_timestep in [0]:
                batch_predictions = get_score(
                    model_dir,
                    model_path,
                    input_noise,
                    model_type,
                    meta_dir=meta_dir,
                    challenge_name="challenge_with_id.csv",
                    batch_size=batch_size,
                    parallel_batch=num_noise_per_time_step,
                    additional_timestep=additional_timestep,
                    timestep=timestep,
                )
                batches_of_predictions.append(batch_predictions)
        all_predictions = torch.cat(batches_of_predictions, dim=-1)

        all_predictions = regression_model(all_predictions).detach().cpu().numpy()
        # clip to [0, 1]
        min_output, max_output = np.min(all_predictions), np.max(all_predictions)
        all_predictions = (all_predictions - min_output) / (max_output - min_output)
        all_predictions = torch.tensor(all_predictions)

        assert torch.all((all_predictions >= 0) & (all_predictions <= 1))
        with open(model_dir / predictions_file_name, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Write each value in a separate row
            for value in list(all_predictions.numpy().squeeze()):
                writer.writerow([value])

    mia_performance_train = evaluate_attack_performance(
        train_indices, "train", model_data_dir, model_type, predictions_file_name
    )
    mia_performance_val = evaluate_attack_performance(
        val_indices, "test", model_data_dir, model_type, predictions_file_name
    )
    mia_performance_test = evaluate_attack_performance(
        test_indices, "final", model_data_dir, model_type, predictions_file_name
    )

    if save_results:
        configure(
            identifier="tartan_federer_attack_results",
            filename=str(results_path / "tartan_federer_attack_results.log"),
        )
        log(INFO, "MIA performance for training set:")
        log(INFO, mia_performance_train)
        log(INFO, "MIA performance for test set:")
        log(INFO, mia_performance_val)
        log(INFO, "MIA performance for final set:")
        log(INFO, mia_performance_test)

    return mia_performance_train, mia_performance_val, mia_performance_test
