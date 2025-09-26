"""Functions in this module are taken with some modifications from CITADEL & UQAM team's attack implementation at
https://github.com/CRCHUM-CITADEL/ensemble-mia.
"""

from typing import Any
from logging import WARNING
import numpy as np
import pandas as pd
import torch
from torch import optim
from midst_toolkit.common.logger import log
from midst_toolkit.models.clavaddpm.gaussian_multinomial_diffusion import (
    GaussianMultinomialDiffusion,
)
from midst_toolkit.models.clavaddpm.model import (
    Transformations,
    Classifier,
    get_model,
    get_model_params,
    get_T_dict,
    get_table_info,
    make_dataset_from_df,
    prepare_fast_dataloader,
)
from midst_toolkit.models.clavaddpm.train import (
    _numerical_forward_backward_log,
)
from midst_toolkit.models.clavaddpm.sampler import (
    create_named_schedule_sampler,
)
from midst_toolkit.models.clavaddpm.trainer import ClavaDDPMTrainer
from midst_toolkit.models.clavaddpm.typing import Configs, RelationOrder, Tables


def fine_tune_model(
    trained_diffusion: GaussianMultinomialDiffusion,
    data_frame: pd.DataFrame,
    data_frame_info: pd.DataFrame,
    model_params: dict[str, Any],
    transformations_dict: dict[str, Any],
    steps: int,
    batch_size: int,
    model_type: str,
    lr: float,
    weight_decay: float,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """
    Fine-tune a trained diffusion model on a new dataset.

    Args:
        trained_diffusion: The pre-trained diffusion model to be fine-tuned.
        data_frame: The new dataset to fine-tune the model on.
        data_frame_info: Information about the new dataset.
        model_params: Parameters for the model architecture.
        transformations_dict: Dictionary containing transformation configurations.
        steps: Number of training steps for fine-tuning.
        batch_size: Batch size for fine-tuning.
        model_type: Type of model architecture to use. mlp or resnet currently supported.
        lr: Learning rate for the optimizer in the diffusion model.
        weight_decay: Weight decay for the diffusion optimizer.
        device: Device to run the training on, either 'cuda' or 'cpu'. Defaults to 'cuda' if available.

    Returns:
        A dictionary containing the fine-tuning results. It will contain the following keys:
            - diffusion: The diffusion model.
            - label_encoders: The label encoders.
            - dataset: The dataset.
            - column_orders: The column orders.
    """
    transformations = Transformations(**transformations_dict)
    # ruff: noqa: N806
    dataset, label_encoders, column_orders = make_dataset_from_df(
        data_frame,
        transformations,
        is_y_cond=model_params["is_y_cond"],
        ratios=[0.99, 0.005, 0.005],
        df_info=data_frame_info,
        std=0,
    )

    category_sizes = np.array(dataset.get_category_sizes("train"))
    if len(category_sizes) == 0 or transformations_dict["cat_encoding"] == "one-hot":
        category_sizes = np.array([0])

    num_numerical_features = dataset.X_num["train"].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(category_sizes) + num_numerical_features
    model_params["d_in"] = d_in

    model = get_model(model_type, model_params)
    model.to(device)

    train_loader = prepare_fast_dataloader(dataset, split="train", batch_size=batch_size)

    diffusion = trained_diffusion
    diffusion.to(device)
    diffusion.train()

    trainer = ClavaDDPMTrainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device,
    )
    trainer.train()

    if model_params["is_y_cond"] == "concat":
        column_orders = column_orders[1:] + [column_orders[0]]
    else:
        column_orders = column_orders + [data_frame_info["y_col"]]

    return {
        "diffusion": diffusion,
        "label_encoders": label_encoders,
        "dataset": dataset,
        "column_orders": column_orders,
    }

# This function will not be called since ensemble is for single-table data, but I am adding it here for completeness
# in case we wanted to experiment with multi-table as well.
def fine_tune_classifier(
    pre_trained_classifier:Classifier,
    data_frame: pd.DataFrame,
    data_frame_info: dict[str, Any],
    model_params: dict[str, Any],
    transformations_dict: dict[str, Any],
    classifier_steps: int,
    batch_size: int,
    gaussian_loss_type: str,
    num_timesteps: int,
    scheduler: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    learning_rate: float = 0.0001,
) -> Classifier:
    """
    Fine-tuning function for the classifier model.

    Args:
        pre_trained_classifier: The pre-trained classifier model to be fine-tuned.
        data_frame: DataFrame to train the model on.
        data_frame_info: Dictionary of the table information.
        model_params: Dictionary of the model parameters.
        transformations_dict: Dictionary of the transformations.
        classifier_steps: Number of steps to fine-tune the classifier.
        batch_size: Batch size to use for training.
        gaussian_loss_type: Type of the gaussian loss to use.
        num_timesteps: Number of timesteps to use for the diffusion model.
        scheduler: Scheduler to use for the diffusion model.
        device: Device to use for training.

    Returns:
        The fine-tuned classifier model.
    """
    transformations = Transformations(**transformations_dict)
    # ruff: noqa: N806
    dataset, label_encoders, column_orders = make_dataset_from_df(
        data_frame,
        transformations,
        is_y_cond=model_params["is_y_cond"],
        ratios=[0.99, 0.005, 0.005],
        df_info=data_frame_info,
        std=0,
    )
    train_loader = prepare_fast_dataloader(
        dataset, split="train", batch_size=batch_size, y_type="long"
    )

    category_sizes = np.array(dataset.get_category_sizes("train"))
    # ruff: noqa: N806
    if len(category_sizes) == 0 or transformations_dict["cat_encoding"] == "one-hot":
        category_sizes = np.array([0])
        # ruff: noqa: N806

    if dataset.X_num is None:
        log(WARNING, "dataset.X_num is None. num_numerical_features will be set to 0")
        num_numerical_features = 0
    else:
        num_numerical_features = dataset.X_num["train"].shape[1]

    if model_params["is_y_cond"] == "concat":
        num_numerical_features -= 1

    classifier = pre_trained_classifier.to(device)

    classifier_optimizer = optim.AdamW(classifier.parameters(), lr=learning_rate)

    empty_diffusion = GaussianMultinomialDiffusion(
        num_classes=category_sizes,
        num_numerical_features=num_numerical_features,
        denoise_fn=None,  # type: ignore[arg-type]
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=torch.device(device),
    )
    empty_diffusion.to(device)

    schedule_sampler = create_named_schedule_sampler("uniform", empty_diffusion)

    classifier.train()
    for step in range(classifier_steps):
        _numerical_forward_backward_log(
            classifier,
            classifier_optimizer,
            train_loader,
            dataset,
            schedule_sampler,
            empty_diffusion,
            prefix="train",
            device=device,
        )

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
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """Fine-tune a child model based on the parent model."""
    if parent_name is None:
        y_col = "placeholder"
        child_df_with_cluster["placeholder"] = list(range(len(child_df_with_cluster)))
    else:
        y_col = f"{parent_name}_{child_name}_cluster"
    child_info = get_table_info(child_df_with_cluster, child_domain_dict, y_col)
    child_model_params = get_model_params(
        {
            "d_layers": diffusion_config["d_layers"],
            "dropout": diffusion_config["dropout"],
        }
    )
    child_t_dict = get_T_dict()

    child_result = fine_tune_model(
        pre_trained_model["diffusion"],
        child_df_with_cluster,
        child_info,
        child_model_params,
        child_t_dict,
        fine_tuning_diffusion_iterations,  # fine_tuning_diffusion_iterations used here.
        diffusion_config["batch_size"],
        diffusion_config["model_type"],
        diffusion_config["lr"],
        diffusion_config["weight_decay"],
        device=device,
    )

    if parent_name is None:
        child_result["classifier"] = None
    else:
        log(WARNING, "Ensemble attack is designed for single table. You are using multi-table fine-tuning.")
        assert (
            classifier_config is not None
        ), "Classifier config is required for multi-table training"
        if classifier_config["iterations"] > 0:
            child_classifier = fine_tune_classifier(
                pre_trained_model["classifier"],
                child_df_with_cluster,
                child_info,
                child_model_params,
                child_t_dict,
                fine_tuning_classifier_iterations,
                classifier_config["batch_size"],
                classifier_config["gaussian_loss_type"],
                classifier_config["num_timesteps"],
                classifier_config["scheduler"],
                device=device,
                lr=classifier_config["lr"],
            )
            child_result["classifier"] = child_classifier
        else:
            log(WARNING, "Skipping classifier training since classifier_config['iterations'] <= 0")

    child_result["df_info"] = child_info
    child_result["model_params"] = child_model_params
    child_result["T_dict"] = child_t_dict
    return child_result


def clava_fine_tuning(
    trained_models: dict[str, Any],
    new_tables: Tables,
    relation_order: RelationOrder,
    diffusion_config: Configs,
    classifier_config: Configs,
    fine_tuning_diffusion_iterations: int,
    fine_tuning_classifier_iterations: int,
) -> dict[tuple[str | None, str], Any]:
    """Fine-tune the trained models on new tables data."""
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
