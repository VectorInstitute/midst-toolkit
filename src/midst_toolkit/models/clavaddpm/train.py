"""Defines the training functions for the ClavaDDPM model."""

import logging
import pickle
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, optim

from midst_toolkit.core import logger
from midst_toolkit.models.clavaddpm.dataset import Dataset, Transformations, get_T_dict, make_dataset_from_df
from midst_toolkit.models.clavaddpm.gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from midst_toolkit.models.clavaddpm.model import (
    Classifier,
    get_model,
    get_table_info,
    prepare_fast_dataloader,
)
from midst_toolkit.models.clavaddpm.sampler import ScheduleSampler, create_named_schedule_sampler
from midst_toolkit.models.clavaddpm.trainer import ClavaDDPMTrainer
from midst_toolkit.models.clavaddpm.typing import Configs, RelationOrder, Tables


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def clava_training(
    tables: Tables,
    relation_order: RelationOrder,
    save_dir: Path,
    diffusion_config: Configs,
    classifier_config: Configs | None,
    device: str = "cuda",
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Training function for the ClavaDDPM model.

    Args:
        tables: Definition of the tables and their relations. Example:
            {
                "table1": {
                    "children": ["table2"],
                    "parents": []
                },
                "table2": {
                    "children": [],
                    "parents": ["table1"]
                }
            }
        relation_order: List of tuples of parent and child tables. Example:
            [("table1", "table2"), ("table1", "table3")]
        save_dir: Directory to save the ClavaDDPM models.
        diffusion_config: Dictionary of configurations for the diffusion model. The following config keys are required:
            {
                d_layers = list[int],
                dropout = float,
                iterations = int,
                batch_size = int,
                model_type = str["mlp" | "resnet"],
                gaussian_loss_type = str["mse" | "cross_entropy"],
                num_timesteps = int,
                scheduler = str["cosine" | "linear"],
                lr = float,
                weight_decay = float,
            }
        classifier_config: Dictionary of configurations for the classifier model. Not required for single table
            training. The following config keys are required for multi-table training:
            {
                iterations = int,
                batch_size = int,
                d_layers = list[int],
                dim_t = int,
                lr = float,
            }
        device: Device to use for training. Default is `"cuda"`.

    Returns:
        Dictionary of models for each parent-child pair.
    """
    models = {}
    for parent, child in relation_order:
        print(f"Training {parent} -> {child} model from scratch")
        df_with_cluster = tables[child]["df"]
        id_cols = [col for col in df_with_cluster.columns if "_id" in col]
        df_without_id = df_with_cluster.drop(columns=id_cols)

        result = child_training(
            df_without_id,
            tables[child]["domain"],
            parent,
            child,
            diffusion_config,
            classifier_config,
            device,
        )

        models[(parent, child)] = result

        target_folder = save_dir / "models"
        target_file = target_folder / f"{parent}_{child}_ckpt.pkl"

        create_message = f"Creating {target_folder}. " if not target_folder.exists() else ""
        LOGGER.info(f"{create_message}Saving {parent} -> {child} model to {target_file}")

        target_folder.mkdir(parents=True, exist_ok=True)
        with open(target_file, "wb") as f:
            pickle.dump(result, f)

    return models


def child_training(
    child_df_with_cluster: pd.DataFrame,
    child_domain_dict: dict[str, Any],
    parent_name: str | None,
    child_name: str,
    diffusion_config: Configs,
    classifier_config: Configs | None,
    device: str = "cuda",
) -> dict[str, Any]:
    """
    Training function for a single child table.

    Args:
        child_df_with_cluster: DataFrame with the cluster column.
        child_domain_dict: Dictionary of the child table domain. It should contain size and type for each
            column of the table. For example:
                {
                    "frequency": {"size": 3, "type": "discrete"},
                    "account_date": {"size": 1535, "type": "continuous"},
                }
        parent_name: Name of the parent table, or None if there is no parent.
        child_name: Name of the child table.
        diffusion_config: Dictionary of configurations for the diffusion model. The following config keys are required:
            {
                d_layers = list[int],
                dropout = float,
                iterations = int,
                batch_size = int,
                model_type = str["mlp" | "resnet"],
                gaussian_loss_type = str["mse" | "cross_entropy"],
                num_timesteps = int,
                scheduler = str["cosine" | "linear"],
                lr = float,
                weight_decay = float,
            }
        classifier_config: Dictionary of configurations for the classifier model. Not required for single table
            training. The following config keys are required for multi-table training:
            {
                iterations = int,
                batch_size = int,
                d_layers = list[int],
                dim_t = int,
                lr = float,
            }
        device: Device to use for training. Default is `"cuda"`.

    Returns:
        Dictionary of the training results.
    """
    if parent_name is None:
        # If there is no parent for this child table, just set a placeholder
        # for its column name. This can happen on single table training or
        # when the table is on the top level of the hierarchy.
        y_col = "placeholder"
        child_df_with_cluster["placeholder"] = list(range(len(child_df_with_cluster)))
    else:
        y_col = f"{parent_name}_{child_name}_cluster"
    child_info = get_table_info(child_df_with_cluster, child_domain_dict, y_col)
    child_model_params = _get_model_params(
        {
            "d_layers": diffusion_config["d_layers"],
            "dropout": diffusion_config["dropout"],
        }
    )
    child_T_dict = get_T_dict()
    # ruff: noqa: N806

    child_result = train_model(
        child_df_with_cluster,
        child_info,
        child_model_params,
        child_T_dict,
        diffusion_config["iterations"],
        diffusion_config["batch_size"],
        diffusion_config["model_type"],
        diffusion_config["gaussian_loss_type"],
        diffusion_config["num_timesteps"],
        diffusion_config["scheduler"],
        diffusion_config["lr"],
        diffusion_config["weight_decay"],
        device=device,
    )

    if parent_name is None:
        child_result["classifier"] = None
    else:
        assert classifier_config is not None, "Classifier config is required for multi-table training"
        if classifier_config["iterations"] > 0:
            child_classifier = train_classifier(
                child_df_with_cluster,
                child_info,
                child_model_params,
                child_T_dict,
                classifier_config["iterations"],
                classifier_config["batch_size"],
                diffusion_config["gaussian_loss_type"],
                diffusion_config["num_timesteps"],
                diffusion_config["scheduler"],
                cluster_col=y_col,
                d_layers=classifier_config["d_layers"],
                dim_t=classifier_config["dim_t"],
                lr=classifier_config["lr"],
                device=device,
            )
            child_result["classifier"] = child_classifier
        else:
            LOGGER.warning("Skipping classifier training since classifier_config['iterations'] <= 0")

    child_result["df_info"] = child_info
    child_result["model_params"] = child_model_params
    child_result["T_dict"] = child_T_dict
    return child_result


def train_model(
    df: pd.DataFrame,
    df_info: pd.DataFrame,
    model_params: dict[str, Any],
    T_dict: dict[str, Any],
    # ruff: noqa: N803
    steps: int,
    batch_size: int,
    model_type: str,
    gaussian_loss_type: str,
    num_timesteps: int,
    scheduler: str,
    lr: float,
    weight_decay: float,
    device: str = "cuda",
) -> dict[str, Any]:
    """
    Training function for the diffusion model.

    Args:
        df: DataFrame to train the model on.
        df_info: Dictionary of the table information.
        model_params: Dictionary of the model parameters.
        T_dict: Dictionary of the transformations.
        steps: Number of steps to train the model.
        batch_size: Batch size to use for training.
        model_type: Type of the model to use.
        gaussian_loss_type: Type of the gaussian loss to use.
        num_timesteps: Number of timesteps to use for the diffusion model.
        scheduler: Scheduler to use for the diffusion model.
        lr: Learning rate to use for the diffusion model.
        weight_decay: Weight decay to use for the diffusion model.
        device: Device to use for training. Default is `"cuda"`.

    Returns:
        Dictionary of the training results. It will contain the following keys:
            - diffusion: The diffusion model.
            - label_encoders: The label encoders.
            - dataset: The dataset.
            - column_orders: The column orders.
    """
    T = Transformations(**T_dict)
    # ruff: noqa: N806
    dataset, label_encoders, column_orders = make_dataset_from_df(
        df,
        T,
        is_y_cond=model_params["is_y_cond"],
        ratios=[0.99, 0.005, 0.005],
        df_info=df_info,
        std=0,
    )

    K = np.array(dataset.get_category_sizes("train"))
    # ruff: noqa: N806
    if len(K) == 0 or T_dict["cat_encoding"] == "one-hot":
        K = np.array([0])
        # ruff: noqa: N806

    num_numerical_features = dataset.X_num["train"].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features
    model_params["d_in"] = d_in

    print("Model params: {}".format(model_params))
    model = get_model(model_type, model_params)
    model.to(device)

    train_loader = prepare_fast_dataloader(dataset, split="train", batch_size=batch_size)

    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=torch.device(device),
    )
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


def train_classifier(
    df: pd.DataFrame,
    df_info: pd.DataFrame,
    model_params: dict[str, Any],
    T_dict: dict[str, Any],
    # ruff: noqa: N803
    classifier_steps: int,
    batch_size: int,
    gaussian_loss_type: str,
    num_timesteps: int,
    scheduler: str,
    d_layers: list[int],
    device: str = "cuda",
    cluster_col: str = "cluster",
    dim_t: int = 128,
    lr: float = 0.0001,
) -> Classifier:
    """
    Training function for the classifier model.

    Args:
        df: DataFrame to train the model on.
        df_info: Dictionary of the table information.
        model_params: Dictionary of the model parameters.
        T_dict: Dictionary of the transformations.
        classifier_steps: Number of steps to train the classifier.
        batch_size: Batch size to use for training.
        gaussian_loss_type: Type of the gaussian loss to use.
        num_timesteps: Number of timesteps to use for the diffusion model.
        scheduler: Scheduler to use for the diffusion model.
        d_layers: List of the hidden sizes of the classifier.
        device: Device to use for training. Default is `"cuda"`.
        cluster_col: Name of the cluster column. Default is `"cluster"`.
        dim_t: Dimension of the timestamp. Default is 128.
        lr: Learning rate to use for the classifier. Default is 0.0001.

    Returns:
        The trained classifier model.
    """
    T = Transformations(**T_dict)
    # ruff: noqa: N806
    dataset, label_encoders, column_orders = make_dataset_from_df(
        df,
        T,
        is_y_cond=model_params["is_y_cond"],
        ratios=[0.99, 0.005, 0.005],
        df_info=df_info,
        std=0,
    )
    print(dataset.n_features)
    train_loader = prepare_fast_dataloader(dataset, split="train", batch_size=batch_size, y_type="long")
    val_loader = prepare_fast_dataloader(dataset, split="val", batch_size=batch_size, y_type="long")
    test_loader = prepare_fast_dataloader(dataset, split="test", batch_size=batch_size, y_type="long")

    eval_interval = 5

    K = np.array(dataset.get_category_sizes("train"))
    # ruff: noqa: N806
    if len(K) == 0 or T_dict["cat_encoding"] == "one-hot":
        K = np.array([0])
        # ruff: noqa: N806
    print(K)

    num_numerical_features = dataset.X_num["train"].shape[1] if dataset.X_num is not None else 0
    if model_params["is_y_cond"] == "concat":
        num_numerical_features -= 1

    classifier = Classifier(
        d_in=num_numerical_features,
        d_out=int(max(df[cluster_col].values) + 1),
        dim_t=dim_t,
        hidden_sizes=d_layers,
    ).to(device)

    classifier_optimizer = optim.AdamW(classifier.parameters(), lr=lr)

    empty_diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
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
    resume_step = 0
    for step in range(classifier_steps):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * batch_size,
        )
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

        classifier_optimizer.step()
        if not step % eval_interval:
            with torch.no_grad():
                classifier.eval()
                _numerical_forward_backward_log(
                    classifier,
                    classifier_optimizer,
                    val_loader,
                    dataset,
                    schedule_sampler,
                    empty_diffusion,
                    prefix="val",
                    device=device,
                )
                classifier.train()

    # # test classifier
    classifier.eval()

    correct = 0
    for _ in range(3000):
        test_x, test_y = next(test_loader)
        test_y = test_y.long().to(device)
        test_x = test_x[:, 1:].to(device) if model_params["is_y_cond"] == "concat" else test_x.to(device)
        with torch.no_grad():
            pred = classifier(test_x, timesteps=torch.zeros(test_x.shape[0]).to(device))
            correct += (pred.argmax(dim=1) == test_y).sum().item()

    acc = correct / (3000 * batch_size)
    print(acc)

    return classifier


def _numerical_forward_backward_log(
    classifier: Classifier,
    optimizer: torch.optim.Optimizer,
    data_loader: Generator[tuple[Tensor, ...]],
    dataset: Dataset,
    schedule_sampler: ScheduleSampler,
    diffusion: GaussianMultinomialDiffusion,
    prefix: str = "train",
    remove_first_col: bool = False,
    device: str = "cuda",
) -> None:
    """
    Forward and backward pass for the numerical features of the ClavaDDPM model.

    Args:
        classifier: The classifier model.
        optimizer: The optimizer.
        data_loader: The data loader.
        dataset: The dataset.
        schedule_sampler: The schedule sampler.
        diffusion: The diffusion object.
        prefix: The prefix for the loss. Defaults to "train".
        remove_first_col: Whether to remove the first column of the batch. Defaults to False.
        device: The device to use. Defaults to "cuda".
    """
    batch, labels = next(data_loader)
    labels = labels.long().to(device)

    if remove_first_col:
        # Remove the first column of the batch, which is the label.
        batch = batch[:, 1:]

    num_batch = batch[:, : dataset.n_num_features].to(device)

    t, _ = schedule_sampler.sample(num_batch.shape[0], device)
    batch = diffusion.gaussian_q_sample(num_batch, t).to(device)

    for i, (sub_batch, sub_labels, sub_t) in enumerate(_split_microbatches(-1, batch, labels, t)):
        logits = classifier(sub_batch, timesteps=sub_t)
        loss = torch.nn.functional.cross_entropy(logits, sub_labels, reduction="none")

        losses = {}
        losses[f"{prefix}_loss"] = loss.detach()
        losses[f"{prefix}_acc@1"] = _compute_top_k(logits, sub_labels, k=1, reduction="none")
        if logits.shape[1] >= 5:
            losses[f"{prefix}_acc@5"] = _compute_top_k(logits, sub_labels, k=5, reduction="none")
        _log_loss_dict(diffusion, sub_t, losses)
        del losses
        loss = loss.mean()
        if loss.requires_grad:
            if i == 0:
                optimizer.zero_grad()
            loss.backward(loss * len(sub_batch) / len(batch))  # type: ignore[no-untyped-call]


def _compute_top_k(logits: Tensor, labels: Tensor, k: int, reduction: str = "mean") -> Tensor:
    """
    Compute the top-k accuracy.

    Args:
        logits: The logits of the classifier.
        labels: The labels of the data.
        k: The number of top-k.
        reduction: The reduction method. Should be one of ["mean", "none"]. Defaults to "mean".

    Returns:
        The top-k accuracy.
    """
    _, top_ks = torch.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean()
    if reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)

    raise ValueError(f"reduction should be one of ['mean', 'none']: {reduction}")


def _log_loss_dict(diffusion: GaussianMultinomialDiffusion, ts: Tensor, losses: dict[str, Tensor]) -> None:
    """
    Output the log loss dictionary in the logger.

    Args:
        diffusion: The diffusion object.
        ts: The timesteps.
        losses: The losses.
    """
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def _split_microbatches(
    microbatch: int,
    batch: Tensor,
    labels: Tensor,
    t: Tensor,
) -> Generator[tuple[Tensor, Tensor, Tensor]]:
    """
    Split the batch into microbatches.

    Args:
        microbatch: The size of the microbatch. If -1, the batch is not split.
        batch: The batch of data as a tensor.
        labels: The labels of the data as a tensor.
        t: The timesteps tensor.

    Returns:
        A generator of for the minibatch which outputs tuples of the batch, labels, and timesteps.
    """
    bs = len(batch)
    if microbatch == -1 or microbatch >= bs:
        yield batch, labels, t
    else:
        for i in range(0, bs, microbatch):
            yield batch[i : i + microbatch], labels[i : i + microbatch], t[i : i + microbatch]


# TODO make this into a class with default parameters
def _get_model_params(rtdl_params: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Return the model parameters.

    Args:
        rtdl_params: The parameters for the RTDL model. If None, the default parameters below are used:
            {
                "d_layers": [512, 1024, 1024, 1024, 1024, 512],
                "dropout": 0.0,
            }

    Returns:
        The model parameters as a dictionary containing the following keys:
            - num_classes: The number of classes. Defaults to 0.
            - is_y_cond: Affects how y is generated. For more information, see the documentation
                of the `make_dataset_from_df` function. Can be any of ["none", "concat", "embedding"].
                Defaults to "none".
            - rtdl_params: The parameters for the RTDL model.
    """
    if rtdl_params is None:
        rtdl_params = {
            "d_layers": [512, 1024, 1024, 1024, 1024, 512],
            "dropout": 0.0,
        }

    return {
        "num_classes": 0,
        "is_y_cond": "none",
        "rtdl_params": rtdl_params,
    }
