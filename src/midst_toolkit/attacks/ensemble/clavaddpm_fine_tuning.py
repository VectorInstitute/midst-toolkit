"""Functions in this module are taken with some modifications from CITADEL & UQAM team's attack implementation at
https://github.com/CRCHUM-CITADEL/ensemble-mia.
TODO: Merge the fine-tuning functionalities in this file with the training functionalities in
`midst_toolkit/models/clavaddpm/train.py`.
"""

from dataclasses import asdict
from logging import WARNING
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import optim

from midst_toolkit.common.enumerations import DataSplit
from midst_toolkit.common.logger import KeyValueLogger, log
from midst_toolkit.common.variables import DEVICE
from midst_toolkit.models.clavaddpm.data_loaders import prepare_fast_dataloader
from midst_toolkit.models.clavaddpm.dataset import (
    Transformations,
    make_dataset_from_df,
)
from midst_toolkit.models.clavaddpm.enumerations import (
    CategoricalEncoding,
    Configs,
    IsTargetConditioned,
    RelationOrder,
    Tables,
    TargetType,
)
from midst_toolkit.models.clavaddpm.gaussian_multinomial_diffusion import (
    GaussianLossType,
    GaussianMultinomialDiffusion,
    SchedulerType,
)
from midst_toolkit.models.clavaddpm.model import (
    Classifier,
    DiffusionParameters,
    ModelParameters,
    get_table_info,
)
from midst_toolkit.models.clavaddpm.sampler import ScheduleSamplerType
from midst_toolkit.models.clavaddpm.train import (
    _numerical_forward_backward_log,
)
from midst_toolkit.models.clavaddpm.trainer import ClavaDDPMTrainer


def fine_tune_model(
    trained_diffusion_model: GaussianMultinomialDiffusion,
    fine_tuning_data: pd.DataFrame,
    fine_tuning_data_info: dict[str, Any],
    model_params: ModelParameters,
    transformations: Transformations,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    data_split_ratios: list[float],
    device: torch.device = DEVICE,
) -> dict[str, Any]:
    """
    Fine-tune a trained diffusion model on a new dataset.

    Args:
        trained_diffusion_model: The pre-trained diffusion model to be fine-tuned.
        fine_tuning_data: The new dataset to fine-tune the model on.
        fine_tuning_data_info: Information about the new dataset.
        model_params: Parameters for the model architecture.
        transformations: Object containing transformation configurations.
        steps: Number of training steps for fine-tuning.
        batch_size: Batch size for fine-tuning.
        lr: Learning rate for the optimizer in the diffusion model.
        weight_decay: Weight decay for the diffusion optimizer.
        data_split_ratios: The ratios of the dataset to split into train, validation, and test.
            It must have exactly 3 values and their sum must amount to 1 (with a tolerance of 0.01).
        device: Device to run the training on, either 'cuda' or 'cpu'. Defaults to 'cuda' if available.

    Returns:
        A dictionary containing the fine-tuning results. It will contain the following keys:
            - diffusion: The diffusion model.
            - label_encoders: The label encoders.
            - dataset: The dataset.
            - column_orders: The column orders.
    """
    dataset, label_encoders, column_orders = make_dataset_from_df(
        fine_tuning_data,
        transformations,
        is_target_conditioned=model_params.is_target_conditioned,
        data_split_ratios=data_split_ratios,
        info=fine_tuning_data_info,
        noise_scale=0,
    )

    category_sizes = np.array(dataset.get_category_sizes(DataSplit.TRAIN))
    if len(category_sizes) == 0 or transformations.categorical_encoding == CategoricalEncoding.ONE_HOT:
        category_sizes = np.array([0])

    num_numerical_features = dataset.x_num[DataSplit.TRAIN.value].shape[1] if dataset.x_num is not None else 0

    train_loader = prepare_fast_dataloader(dataset, split=DataSplit.TRAIN, batch_size=batch_size)

    diffusion = trained_diffusion_model
    diffusion.to(device)
    diffusion.train()

    trainer = ClavaDDPMTrainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=str(device),
    )
    trainer.train()

    if model_params.is_target_conditioned == IsTargetConditioned.CONCAT:
        column_orders = column_orders[1:] + [column_orders[0]]
    else:
        column_orders = column_orders + [fine_tuning_data_info["y_col"]]

    return {
        "diffusion": diffusion,
        "label_encoders": label_encoders,
        "dataset": dataset,
        "column_orders": column_orders,
        "num_numerical_features": num_numerical_features,
        "K": category_sizes,
        "is_regression": dataset.is_regression,
        "inverse_transform": (
            dataset.numerical_transform.inverse_transform if dataset.numerical_transform is not None else None
        ),
    }


# NOTE: This function will not be called in the Ensemble attack since Ensemble only covers the single-table setting,
# but this is added here for completeness in case we decide to experiment with multi-table as well.
def fine_tune_classifier(
    pre_trained_classifier: Classifier,
    fine_tuning_data: pd.DataFrame,
    fine_tuning_data_info: dict[str, Any],
    model_params: ModelParameters,
    transformations: Transformations,
    classifier_steps: int,
    batch_size: int,
    gaussian_loss_type: GaussianLossType,
    num_timesteps: int,
    scheduler_type: SchedulerType,
    data_split_ratios: list[float],
    learning_rate: float = 0.0001,
    device: torch.device = DEVICE,
) -> Classifier:
    """
    Fine-tuning function for the classifier model.

    Args:
        pre_trained_classifier: The pre-trained classifier model to be fine-tuned.
        fine_tuning_data: DataFrame to train the model on.
        fine_tuning_data_info: Dictionary of the table information.
        model_params: Parameters for the model architecture.
        transformations: Transformation object containing all the transformations.
        classifier_steps: Number of steps to fine-tune the classifier.
        batch_size: Batch size to use for training.
        gaussian_loss_type: Type of the gaussian loss to use.
        num_timesteps: Number of timesteps to use for the diffusion model.
        scheduler_type: Type of scheduler to use for the diffusion model.
        data_split_ratios: The ratios of the dataset to split into train, validation, and test.
            It must have exactly 3 values and their sum must amount to 1 (with a tolerance of 0.01).
        learning_rate: Learning rate for the optimizer. Default is 0.0001.
        device: Device to use for training.

    Returns:
        The fine-tuned classifier model.
    """
    dataset, label_encoders, column_orders = make_dataset_from_df(
        fine_tuning_data,
        transformations,
        is_target_conditioned=model_params.is_target_conditioned,
        data_split_ratios=data_split_ratios,
        info=fine_tuning_data_info,
        noise_scale=0,
    )
    train_loader = prepare_fast_dataloader(
        dataset,
        split=DataSplit.TRAIN,
        batch_size=batch_size,
        target_type=TargetType.LONG,
    )
    category_sizes = np.array(dataset.get_category_sizes(DataSplit.TRAIN))
    if len(category_sizes) == 0 or transformations.categorical_encoding == CategoricalEncoding.ONE_HOT:
        category_sizes = np.array([0])

    if dataset.x_num is None:
        log(WARNING, "dataset.x_num is None. num_numerical_features will be set to 0")
        num_numerical_features = 0
    else:
        num_numerical_features = dataset.x_num[DataSplit.TRAIN.value].shape[1]

    if model_params.is_target_conditioned == IsTargetConditioned.CONCAT:
        num_numerical_features -= 1

    classifier = pre_trained_classifier.to(device)

    classifier_optimizer = optim.AdamW(classifier.parameters(), lr=learning_rate)

    diffusion = GaussianMultinomialDiffusion(
        num_classes=category_sizes,
        num_numerical_features=num_numerical_features,
        denoise_fn=None,  # type: ignore[arg-type]
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler_type=scheduler_type,
        device=torch.device(device),
    )
    diffusion.to(device)

    schedule_sampler = ScheduleSamplerType.UNIFORM.create_named_schedule_sampler(num_timesteps)
    key_value_logger = KeyValueLogger()
    classifier.train()
    for step in range(classifier_steps):
        key_value_logger.save_entry("step", float(step))
        key_value_logger.save_entry("samples", float((step + 1) * batch_size))
        _numerical_forward_backward_log(
            classifier,
            classifier_optimizer,
            train_loader,
            dataset,
            schedule_sampler,
            diffusion,
            prefix=DataSplit.TRAIN.value,
            device=str(device),
            key_value_logger=key_value_logger,
        )
        # Dump the contents of the key value logger before returning.
        key_value_logger.dump()

    return classifier


def child_fine_tuning(
    pre_trained_model: dict[str, Any],
    child_df_with_cluster: pd.DataFrame,
    child_domain_dict: dict[str, Any],
    parent_name: str | None,
    child_name: str,
    diffusion_config: Configs,
    classifier_config: Configs | None,
    fine_tuning_diffusion_iterations: int,
    fine_tuning_classifier_iterations: int,
    device: torch.device = DEVICE,
) -> dict[str, Any]:
    """
    Fine-tune a child model based on the parent model.

    Args:
        pre_trained_model: The pre-trained model to be fine-tuned.
        child_df_with_cluster: The DataFrame containing the child data with cluster information.
        child_domain_dict: The domain dictionary for the child data.
        parent_name: The name of the parent table. None if the child is the root table.
        child_name: The name of the child table.
        diffusion_config: The configuration for the diffusion model.
        classifier_config: The configuration for the classifier model. None if no classifier is used.
        fine_tuning_diffusion_iterations: The number of iterations for fine-tuning the diffusion model.
        fine_tuning_classifier_iterations: The number of iterations for fine-tuning the classifier model.
        device: The device to run the fine-tuning on. Defaults to 'cuda' if available.

    Returns:
        A dictionary containing the fine-tuned model and related information.

    """
    if parent_name is None:
        target_col = "placeholder"
        child_df_with_cluster["placeholder"] = list(range(len(child_df_with_cluster)))
    else:
        target_col = f"{parent_name}_{child_name}_cluster"
    child_info = get_table_info(child_df_with_cluster, child_domain_dict, target_col)
    child_model_params = ModelParameters(
        diffusion_parameters=DiffusionParameters(
            d_layers=diffusion_config["d_layers"],
            dropout=diffusion_config["dropout"],
        ),
    )
    child_transformations = Transformations.default()

    child_result = fine_tune_model(
        pre_trained_model["diffusion"],
        child_df_with_cluster,
        child_info,
        child_model_params,
        child_transformations,
        fine_tuning_diffusion_iterations,
        diffusion_config["batch_size"],
        diffusion_config["lr"],
        diffusion_config["weight_decay"],
        diffusion_config["data_split_ratios"],
        device=device,
    )

    if parent_name is None:
        child_result["classifier"] = None
    else:
        log(
            WARNING,
            "Ensemble attack is designed for single table. You are using multi-table fine-tuning.",
        )
        assert classifier_config is not None, "Classifier config is required for multi-table training"
        if classifier_config["iterations"] > 0:
            child_classifier = fine_tune_classifier(
                pre_trained_model["classifier"],
                child_df_with_cluster,
                child_info,
                child_model_params,
                child_transformations,
                fine_tuning_classifier_iterations,
                classifier_config["batch_size"],
                GaussianLossType(diffusion_config["gaussian_loss_type"]),
                classifier_config["num_timesteps"],
                SchedulerType(diffusion_config["scheduler"]),
                data_split_ratios=classifier_config["data_split_ratios"],
                learning_rate=classifier_config["lr"],
                device=device,
            )
            child_result["classifier"] = child_classifier
        else:
            log(
                WARNING,
                "Skipping classifier training since classifier_config['iterations'] <= 0",
            )

    child_result["df_info"] = child_info
    child_result["model_params"] = asdict(child_model_params)
    child_result["T_dict"] = asdict(child_transformations)
    return child_result


def clava_fine_tuning(
    trained_models: dict[tuple[str, str], dict[str, Any]],
    new_tables: Tables,
    relation_order: RelationOrder,
    diffusion_config: Configs,
    classifier_config: Configs,
    fine_tuning_diffusion_iterations: int,
    fine_tuning_classifier_iterations: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Fine-tune the trained models on new tables data.

    Args:
        trained_models: The previously trained model material.
        new_tables: The new tables data to fine-tune the models on.
        relation_order: The relation order of the tables.
        diffusion_config: The configuration for the diffusion model.
        classifier_config: The configuration for the classifier model.
        fine_tuning_diffusion_iterations: The number of iterations for fine-tuning the diffusion model.
        fine_tuning_classifier_iterations: The number of iterations for fine-tuning the classifier model.

    Returns:
        A dictionary containing the fine-tuned models for each (parent, child) table pair.

    """
    new_models = {}
    for parent, child in relation_order:
        df_with_cluster = new_tables[child]["df"]
        id_cols = [col for col in df_with_cluster.columns if "_id" in col]
        df_without_id = df_with_cluster.drop(columns=id_cols)
        child_model = trained_models[(parent, child)]
        result = child_fine_tuning(
            child_model,
            df_without_id,
            new_tables[child]["domain"],
            parent,
            child,
            diffusion_config,
            classifier_config,
            fine_tuning_diffusion_iterations,
            fine_tuning_classifier_iterations,
        )
        new_models[(parent, child)] = result

    return new_models
