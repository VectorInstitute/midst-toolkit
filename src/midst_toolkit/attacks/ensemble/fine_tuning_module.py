"""Functions in this module are taken with some modifications from CITADEL & UQAM team's attack implementation at
https://github.com/CRCHUM-CITADEL/ensemble-mia.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch

from midst_toolkit.models.clavaddpm.gaussian_multinomial_diffusion import (
    GaussianMultinomialDiffusion,
)
from midst_toolkit.models.clavaddpm.model import (
    Transformations,
    get_model,
    get_model_params,
    get_T_dict,
    get_table_info,
    make_dataset_from_df,
    prepare_fast_dataloader,
)
from midst_toolkit.models.clavaddpm.train import train_classifier
from midst_toolkit.models.clavaddpm.trainer import ClavaDDPMTrainer


def fine_tune_model(
    trained_diffusion: GaussianMultinomialDiffusion,
    df: pd.DataFrame,
    df_info: pd.DataFrame,
    model_params: dict[str, Any],
    t_dict: dict[str, Any],
    steps: int,
    batch_size: int,
    model_type: str,
    lr: float,
    weight_decay: float,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """Fine-tune a a trained diffusion model on a new dataset."""
    transformations = Transformations(**t_dict)
    dataset, label_encoders, column_orders = make_dataset_from_df(
        df,
        transformations,
        is_y_cond=model_params["is_y_cond"],
        ratios=[0.99, 0.005, 0.005],
        df_info=df_info,
        std=0,
    )
    train_loader = prepare_fast_dataloader(dataset, split="train", batch_size=batch_size, y_type="long")

    num_numerical_features = dataset.X_num["train"].shape[1] if dataset.X_num is not None else 0

    category_array = np.array(dataset.get_category_sizes("train"))
    if len(category_array) == 0 or t_dict["cat_encoding"] == "one-hot":
        category_array = np.array([0])

    num_numerical_features = dataset.X_num["train"].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(category_array) + num_numerical_features
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
    fine_tuning_diffusion_iterations: int,
    fine_tuning_classifier_iterations: int,
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
            "d_layers": configs["diffusion"]["d_layers"],
            "dropout": configs["diffusion"]["dropout"],
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
        configs["diffusion"]["batch_size"],
        configs["diffusion"]["model_type"],
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
            child_t_dict,
            fine_tuning_classifier_iterations,  # fine_tuning_classifier_iterations used here.
            configs["classifier"]["batch_size"],
            configs["diffusion"]["gaussian_loss_type"],
            configs["diffusion"]["num_timesteps"],
            configs["diffusion"]["scheduler"],
            cluster_col=y_col,
            d_layers=configs["classifier"]["d_layers"],
            dim_t=configs["classifier"]["dim_t"],
            lr=configs["classifier"]["lr"],
            pre_trained_classifier=pre_trained_model["classifier"],
        )
        child_result["classifier"] = child_classifier

    child_result["df_info"] = child_info
    child_result["model_params"] = child_model_params
    child_result["T_dict"] = child_t_dict
    return child_result


def clava_fine_tuning(
    trained_models: dict[str, Any],
    new_tables: dict[str, Any],
    relation_order: dict[Any, Any],
    configs: dict[str, Any],
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
            configs,
            fine_tuning_diffusion_iterations,
            fine_tuning_classifier_iterations,
        )
        new_models[(parent, child)] = result

    return new_models
