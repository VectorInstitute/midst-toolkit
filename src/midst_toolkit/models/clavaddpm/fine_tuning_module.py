"""Functions in this module are taken with some modifications from CITADEL & UQAM team's attack implementation at
https://github.com/CRCHUM-CITADEL/ensemble-mia.
"""

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from midst_toolkit.models.clavaddpm.data_loaders import prepare_fast_dataloader
from midst_toolkit.models.clavaddpm.dataset import make_dataset_from_df
from midst_toolkit.models.clavaddpm.gaussian_multinomial_diffusion import (
    GaussianMultinomialDiffusion,
)
from midst_toolkit.models.clavaddpm.model import ModelType, get_table_info
from midst_toolkit.models.clavaddpm.train import train_classifier
from midst_toolkit.models.clavaddpm.trainer import ClavaDDPMTrainer
from midst_toolkit.models.clavaddpm.typing import (
    CatEncoding,
    GaussianLossType,
    IsYCond,
    ModelParameters,
    RTDLParameters,
    Scheduler,
    Transformations,
)


def fine_tune_model(
    trained_diffusion: GaussianMultinomialDiffusion,
    df: pd.DataFrame,
    df_info: dict[str, Any],
    model_params: ModelParameters,
    transformations: Transformations,
    steps: int,
    batch_size: int,
    model_type: ModelType,
    lr: float,
    weight_decay: float,
    device: str = "cuda",
) -> dict[str, Any]:
    """Fine-tune a a trained diffusion model on a new dataset."""
    dataset, label_encoders, column_orders = make_dataset_from_df(
        df,
        transformations,
        is_y_cond=model_params.is_y_cond,
        ratios=[0.99, 0.005, 0.005],
        df_info=df_info,
        std=0,
    )
    train_loader = prepare_fast_dataloader(dataset, split="train", batch_size=batch_size, y_type="long")

    num_numerical_features = dataset.X_num["train"].shape[1] if dataset.X_num is not None else 0

    category_array = np.array(dataset.get_category_sizes("train"))
    if len(category_array) == 0 or transformations.cat_encoding == CatEncoding.ONE_HOT:
        category_array = np.array([0])

    num_numerical_features = dataset.X_num["train"].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(category_array) + num_numerical_features
    model_params.d_in = d_in

    model = model_type.get_model(model_params)
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

    if model_params.is_y_cond == IsYCond.CONCAT:
        column_orders = column_orders[1:] + [column_orders[0]]
    else:
        column_orders = column_orders + [df_info["y_col"]]

    return {
        "diffusion": diffusion,
        "label_encoders": label_encoders,
        "dataset": dataset,
        "column_orders": column_orders,
    }


def child_fine_tuning(
    pre_trained_model: dict[str, Any],
    child_df_with_cluster: pd.DataFrame,
    child_domain_dict: dict[str, Any],
    parent_name: str | None,
    child_name: str,
    configs: dict[str, Any],
    new_diffusion_iterations: int,
    new_classifier_iterations: int,
) -> dict[str, Any]:
    """Fine-tune a child model based on the parent model."""
    if parent_name is None:
        y_col = "placeholder"
        child_df_with_cluster["placeholder"] = list(range(len(child_df_with_cluster)))
    else:
        y_col = f"{parent_name}_{child_name}_cluster"
    child_info = get_table_info(child_df_with_cluster, child_domain_dict, y_col)
    child_model_params = ModelParameters(
        rtdl_parameters=RTDLParameters(
            d_layers=configs["diffusion"]["d_layers"],
            dropout=configs["diffusion"]["dropout"],
        ),
    )
    child_transformations = Transformations.default()

    child_result = fine_tune_model(
        pre_trained_model["diffusion"],
        child_df_with_cluster,
        child_info,
        child_model_params,
        child_transformations,
        new_diffusion_iterations,  # new_diffusion_iterations used here.
        configs["diffusion"]["batch_size"],
        ModelType(configs["diffusion"]["model_type"]),
        configs["diffusion"]["lr"],
        configs["diffusion"]["weight_decay"],
    )

    if parent_name is None:
        child_result["classifier"] = None
    elif configs["classifier"]["iterations"] > 0:
        child_classifier = train_classifier(
            child_df_with_cluster,
            child_info,
            child_model_params,
            child_transformations,
            new_classifier_iterations,  # new_classifier_iterations used here.
            configs["classifier"]["batch_size"],
            GaussianLossType(configs["diffusion"]["gaussian_loss_type"]),
            configs["diffusion"]["num_timesteps"],
            Scheduler(configs["diffusion"]["scheduler"]),
            cluster_col=y_col,
            d_layers=configs["classifier"]["d_layers"],
            dim_t=configs["classifier"]["dim_t"],
            learning_rate=configs["classifier"]["lr"],
            pre_trained_classifier=pre_trained_model["classifier"],
        )
        child_result["classifier"] = child_classifier

    child_result["df_info"] = child_info
    child_result["model_params"] = child_model_params
    child_result["T_dict"] = asdict(child_transformations)
    return child_result


def clava_fine_tuning(
    trained_models: dict[str, Any],
    new_tables: dict[str, Any],
    relation_order: dict[Any, Any],
    configs: dict[str, Any],
    new_diffusion_iterations: int,
    new_classifier_iterations: int,
) -> dict[tuple[str | None, str], Any]:
    """Fine-tune the trained models on new tables data."""
    new_models = {}
    for parent, child in relation_order:
        df_with_cluster = new_tables[child]["df"]
        id_cols = [col for col in df_with_cluster.columns if "_id" in col]
        df_without_id = df_with_cluster.drop(columns=id_cols)
        result = child_fine_tuning(
            trained_models,
            df_without_id,
            new_tables[child]["domain"],
            parent,
            child,
            configs,
            new_diffusion_iterations,
            new_classifier_iterations,
        )
        new_models[(parent, child)] = result

    return new_models
