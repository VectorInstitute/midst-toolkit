# ruff: noqa: D102, D105, D103, D200, PLR0915
# mypy: disable-error-code=no-untyped-def
# mypy: disable-error-code=has-type
# mypy: disable-error-code=index
# mypy: disable-error-code=attr-defined
# mypy: disable-error-code=assignment
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from midst_toolkit.attacks.tf.classifcation import MLP, fitmodel
from midst_toolkit.attacks.tf.data_utils import (
    CustomUnpickler,
    TaskType,
    evaluate_attack_performance,
    load_multi_table_customized,
    prepare_data_for_attack,
    prepare_fast_dataloader,
)
from midst_toolkit.models.clavaddpm.dataset import Dataset


# In[34]:


# we define the loss function here
# global variables are assigned in the main process
def mixed_loss(
    diffusion,
    x,
    out_dict,
    noise=None,
    t=None,
    return_random=False,
    no_mean=False,
    parallel_batch=None,
    addt_value=None,
):
    x_num = x[:, : diffusion.num_numerical_features]
    x_cat = x[:, diffusion.num_numerical_features :]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_tensor = torch.tensor(noise, device=device, dtype=torch.float)
    batch_noise = noise_tensor.repeat(x_num.shape[0], 1)

    # there is actually no categorical classes, as we have examined the DM, so we just ignore x_cat here and later
    x_num = x_num.repeat_interleave(parallel_batch, dim=0)
    x_cat = x_cat.repeat_interleave(parallel_batch, dim=0)

    b = x_num.shape[0]

    device = x.device
    if t is None:
        # the defeualt is uniform sampling
        t, pt = diffusion.sample_time(b, device)

    if return_random:
        return noise, t, pt

    additional_t = t * 0 + addt_value

    # forward x_num_t with (t+additional_t) timestamps
    x_num_t = diffusion.gaussian_q_sample(x_num, t + additional_t, noise=batch_noise)

    if not return_random:
        current_t = t
        # predict noises with t timestamps
        predicted_noise = diffusion._denoise_fn(x_num_t, current_t, **out_dict)
        current_loss = diffusion._gaussian_loss(predicted_noise, batch_noise, batch_noise, current_t, batch_noise)
        transformed_current_loss = current_loss.reshape(-1, parallel_batch)

    return transformed_current_loss * 0, transformed_current_loss


def raise_unknown(unknown_what: str, unknown_value: Any):
    raise ValueError(f"Unknown {unknown_what}: {unknown_value}")


def build_target(
    y: dict[str, np.ndarray], policy: Literal["default"] | None, task_type: TaskType
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    info: dict[str, Any] = {"policy": policy}
    if policy is None:
        pass
    elif policy == "default":
        if task_type == TaskType.REGRESSION:
            mean, std = float(y["train"].mean()), float(y["train"].std())
            y = {k: (v - mean) / std for k, v in y.items()}
            info["mean"] = mean
            info["std"] = std
    else:
        raise_unknown("policy", policy)
    return y, info


@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Literal["standard", "quantile", "minmax"] | None = None
    num_nan_policy: Literal["drop-rows", "mean"] | None = None
    cat_nan_policy: Literal["most_frequent"] | None = None
    cat_min_frequency: float | None = None
    cat_encoding: Literal["one-hot", "counter"] | None = None
    y_policy: Literal["default"] | None = "default"


def get_dataset(data_path, target_model_dir=None, train_name="train_with_id.csv", batch_size=None):
    tables, relation_order, _ = load_multi_table_customized(
        data_path,
        meta_dir="/h/behnzaman/midst-experiments/deps/TF_attack/midst_models/single_table_TabDDPM/configs",
        train_name=train_name,
    )
    if len(relation_order) == 1:
        parent, child = relation_order[0]

        df_with_id = tables[child]["df"]
        df_without_id = df_with_id.drop(columns=[col for col in df_with_id.columns if "_id" in col])

        file_path = os.path.join(target_model_dir, f"{parent}_{child}_ckpt.pkl")

        with open(file_path, "rb") as f:
            model = CustomUnpickler(f).load()

        df_without_id["placeholder"] = df_without_id.index
        transformations = model.transformations

        dataset, label_encoders, column_orders = Dataset.from_df(
            data=df_without_id,
            transformations=transformations,
            is_target_conditioned=model.model_parameters.is_target_conditioned,
            data_split_percentages=None,  # manually set test split below
            table_metadata=model.table_metadata,
        )

        train_loader_list = []
        # === Create a "test" split identical to "train" ====
        dataset.numerical_features["train"] = np.concatenate(
            [
                dataset.numerical_features["train"],
                dataset.numerical_features["val"],
                dataset.numerical_features["test"],
            ],
            axis=0,
        )

        if dataset.categorical_features is not None:
            dataset.categorical_features["train"] = np.concatenate(
                [
                    dataset.categorical_features["train"],
                    dataset.categorical_features["val"],
                    dataset.categorical_features["test"],
                ],
                axis=0,
            )

        dataset.target["train"] = np.concatenate(
            [dataset.target["train"], dataset.target["val"], dataset.target["test"]], axis=0
        )

        dataset.numerical_features["test"] = dataset.numerical_features["train"].copy()

        if dataset.categorical_features is not None:
            dataset.categorical_features["test"] = dataset.categorical_features["train"].copy()

        dataset.target["test"] = dataset.target["train"].copy()

        # === Build DataLoader ================================
        train_loader = prepare_fast_dataloader(dataset, split="test", batch_size=batch_size, y_type="long")

        train_loader_list.append([train_loader, dataset.numerical_features["test"].shape[0], dataset])

    else:
        raise NotImplementedError("Only single table datasets are supported.")

    return train_loader_list


def get_score(
    data_path,
    save_dir,
    input_noise,
    type="tabddpm",
    phase=None,
    challenge_name=None,
    batch_size=None,
    parallel_batch=None,
    addt_value=None,
    t_value=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if type == "tabddpm":
        relation_order = [("None", "trans")]
    elif type == "tabsyn":
        raise ValueError("Haven't done it yet!")

    train_loader_list = get_dataset(data_path, save_dir, train_name=challenge_name, batch_size=batch_size)

    """
    Computes the score for a given dataset using a diffusion model.

    Args:
        data_path (str): Path to the dataset.
        save_dir (str): Directory where model checkpoints are saved.
        input_noise (torch.Tensor): Noise tensor to be used in the loss computation.
        config_path (str, optional): Path to the configuration file. Defaults to None.
        type (str, optional): Type of model to use. Defaults to "tabddpm".
        phase (str, optional): Phase of the dataset (e.g., train, test). Defaults to None.
        challenge_name (str, optional): Name of the challenge dataset. Defaults to None.
        batch_size (int, optional): Batch size for data loading. Defaults to None.
        parallel_batch (int, optional): Number of parallel batches for processing. Defaults to None.
        addt_value (Any, optional): Additional value to be passed to the loss function. Defaults to None.
        t_value (Any, optional): Value of the time step `t` to be used in the loss computation. Defaults to None.

    Returns:
        torch.Tensor: A tensor containing the computed loss values.

    Raises:
        ValueError: If the specified `type` is not supported.
        AssertionError: If required model checkpoint files are not found or if `iter_max` is not equal to 1.
    """

    # for tabddpm, relation order only contains like None_trans
    loader_count = 0

    for parent, child in relation_order:
        assert os.path.exists(os.path.join(save_dir, f"{parent}_{child}_ckpt.pkl"))
        train_loader, iter_max, challenge_dataset = train_loader_list[loader_count]

        filepath = os.path.join(save_dir, f"{parent}_{child}_ckpt.pkl")

        # get the diffusion model
        with open(filepath, "rb") as f:
            model = CustomUnpickler(f).load()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        diffusion = model.diffusion.to(device)

        iter_id = 0
        print(iter_max)
        iter_max = iter_max // batch_size
        return_res = torch.zeros([batch_size, parallel_batch])
        assert iter_max == 1
        iter_id = 0
        while iter_id < iter_max:
            x, out_dict = next(train_loader)
            out_dict = {"y": out_dict}
            x = x.to(device)
            for k in out_dict:
                out_dict[k] = out_dict[k].long().to(device)

            with torch.no_grad():
                # get loss here
                noise, t_cur, pt = mixed_loss(
                    diffusion,
                    x,
                    out_dict,
                    noise=input_noise,
                    return_random=True,
                    parallel_batch=parallel_batch,
                    addt_value=addt_value,
                )
                t = t_cur * 0 + t_value
                _, loss = mixed_loss(
                    diffusion,
                    x,
                    out_dict,
                    t=t,
                    noise=input_noise,
                    no_mean=True,
                    parallel_batch=parallel_batch,
                    addt_value=addt_value,
                )

            return_res = loss
            iter_id += 1
    return return_res


def tf_attack_train_classifier(
    train_indices,
    val_indices,
    samples_per_train_model,
    sample_per_val_model,
    model_type,
    tabddpm_data_dir,
    base_path,
    target_model_subdir,
    classifier_hidden_dim,
    use_best_checkpoint,
    num_noise_per_time_step,
    timesteps_list,
    addt_value_list,
    classifier_num_epochs,
    results_path,
    input_noise,
):
    df_train_merge, _, _ = prepare_data_for_attack(
        indices=train_indices,
        model_type=model_type,
        models_base_dir=base_path,
        keys_for_deduplication=["trans_id", "balance"],
    )

    df_test_merge, _, _ = prepare_data_for_attack(
        indices=val_indices,
        model_type=model_type,
        models_base_dir=base_path,
        keys_for_deduplication=["trans_id", "balance"],
    )

    total_data_num_for_train = samples_per_train_model * 2 * len(train_indices)
    x_train = np.zeros([total_data_num_for_train, len(input_noise) * len(timesteps_list) * len(addt_value_list)])
    x_train_label = np.zeros([total_data_num_for_train])

    if val_indices:
        total_data_num_for_validation = sample_per_val_model * 2 * len(val_indices)
        x_val = np.zeros(
            [total_data_num_for_validation, len(input_noise) * len(timesteps_list) * len(addt_value_list)]
        )
        x_val_label = np.zeros([total_data_num_for_validation])
    else:
        x_val, x_val_label = None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regression_model = MLP(
        input_dim=len(input_noise) * len(timesteps_list) * len(addt_value_list), hidden_dim=classifier_hidden_dim
    ).to(device)

    train_count = 0
    val_count = 0
    if val_indices is None:
        val_indices = []
    model_folders_indices = np.concatenate((train_indices, val_indices))
    for model_number in tqdm(model_folders_indices, desc="Processing models", unit="model"):
        model_folder = f"{model_type}_{model_number}"

        model_dir = tabddpm_data_dir / model_folder
        model_path = model_dir / target_model_subdir

        if model_number in train_indices:
            df_train = pd.read_csv(os.path.join(model_dir, "train_with_id.csv"))

            df_exclusive = df_train_merge[
                ~df_train_merge.set_index(["trans_id", "balance"]).index.isin(
                    df_train.set_index(["trans_id", "balance"]).index
                )
            ]

            data_exclusive = df_exclusive.sample(samples_per_train_model)
            data_from_train = df_train.sample(samples_per_train_model)

            df_data = pd.concat([data_exclusive, data_from_train], ignore_index=True)
            df_data.to_csv(os.path.join(model_dir, "data_for_training_MIA.csv"), index=False)

            df_train_merge = df_train_merge[
                ~df_train_merge.set_index(["trans_id", "balance"]).index.isin(
                    df_data.set_index(["trans_id", "balance"]).index
                )
            ]

        elif model_number in val_indices:
            df_test = pd.read_csv(os.path.join(model_dir, "train_with_id.csv"))
            df_exclusive = df_test_merge[
                ~df_test_merge.set_index(["trans_id", "balance"]).index.isin(
                    df_test.set_index(["trans_id", "balance"]).index
                )
            ]

            data_test_exclusive = df_exclusive.sample(sample_per_val_model)
            data_from_test = df_test.sample(sample_per_val_model)

            df_test_data = pd.concat([data_test_exclusive, data_from_test], ignore_index=True)
            df_test_data.to_csv(os.path.join(model_dir, "data_for_validating_MIA.csv"), index=False)

            df_test_merge = df_test_merge[
                ~df_test_merge.set_index(["trans_id", "balance"]).index.isin(
                    df_test_data.set_index(["trans_id", "balance"]).index
                )
            ]

        t_value_count = 0
        for t_value in timesteps_list:
            for addt_value in [0]:
                if model_number in train_indices:
                    batch_size = samples_per_train_model * 2

                    predictions = get_score(
                        model_dir,
                        model_path,
                        input_noise,
                        model_type,
                        phase="train",
                        challenge_name="data_for_training_MIA.csv",
                        batch_size=batch_size,
                        parallel_batch=num_noise_per_time_step,
                        addt_value=addt_value,
                        t_value=t_value,
                    )

                    x_train[
                        samples_per_train_model * 2 * train_count : samples_per_train_model * 2 * (train_count + 1),
                        t_value_count * num_noise_per_time_step : (t_value_count + 1) * num_noise_per_time_step,
                    ] = predictions.detach().squeeze().cpu().numpy()

                    x_train_label[
                        samples_per_train_model * 2 * train_count : samples_per_train_model * 2 * (train_count + 1)
                    ] = np.concatenate([np.zeros(samples_per_train_model), np.ones(samples_per_train_model)])
                    t_value_count += 1

                elif model_number in val_indices:
                    batch_size = sample_per_val_model * 2
                    predictions = get_score(
                        model_dir,
                        model_path,
                        input_noise,
                        model_type,
                        phase="train",
                        challenge_name="data_for_validating_MIA.csv",
                        batch_size=batch_size,
                        parallel_batch=num_noise_per_time_step,
                        addt_value=addt_value,
                        t_value=t_value,
                    )
                    x_val[
                        sample_per_val_model * 2 * val_count : sample_per_val_model * 2 * (val_count + 1),
                        t_value_count * num_noise_per_time_step : (t_value_count + 1) * num_noise_per_time_step,
                    ] = predictions.detach().squeeze().cpu().numpy()

                    x_val_label[sample_per_val_model * 2 * val_count : sample_per_val_model * 2 * (val_count + 1)] = (
                        np.concatenate([np.zeros(sample_per_val_model), np.ones(sample_per_val_model)])
                    )
                    t_value_count += 1

            if model_number in train_indices:
                train_count += 1
            elif model_number in val_indices:
                val_count += 1

    return fitmodel(
        regression_model,
        x_train,
        x_train_label,
        x_val,
        x_val_label,
        num_epochs=classifier_num_epochs,
        use_best_checkpoint=use_best_checkpoint,
        best_model_dir=results_path,
    )


def tf_attack(
    train_indices: list[int],
    val_indices: list[int] | None,
    num_noise_per_time_step: int,
    samples_per_train_model: int,
    sample_per_val_model: int,
    tabddpm_data_dir: Path,
    model_type: str,
    target_model_subdir: str,
    timesteps_list: list[int],
    use_best_checkpoint: bool,
    results_path: str,
    test_indices: list[int],
    predictions_file_format: str,
    base_path: Path,
    classifier_num_epochs: int,
    classifier_hidden_dim: int,
    addt_value_list: list[int],
) -> tuple[Any, Any, Any]:
    os.makedirs(results_path, exist_ok=True)
    input_noise: list[list[float]] = [np.random.normal(size=8).tolist() for _ in range(num_noise_per_time_step)]

    regression_model = tf_attack_train_classifier(
        train_indices,
        val_indices,
        samples_per_train_model,
        sample_per_val_model,
        model_type,
        tabddpm_data_dir,
        base_path,
        target_model_subdir,
        classifier_hidden_dim,
        use_best_checkpoint,
        num_noise_per_time_step,
        timesteps_list,
        addt_value_list,
        classifier_num_epochs,
        results_path,
        input_noise,
    )

    predictions_file_name: str = f"{predictions_file_format}.csv"
    if val_indices is None:
        val_indices = []
    model_folders_indices = np.concatenate((train_indices, val_indices, test_indices))
    for model_number in tqdm(model_folders_indices, desc="Processing models", unit="model"):
        model_folder: str = f"{model_type}_{model_number}"

        model_dir = tabddpm_data_dir / model_folder
        model_path = model_dir / target_model_subdir
        batch_size = 200
        t_value_count = 0
        current_input = []
        for t_value in timesteps_list:
            for addt_value in [0]:
                predictions: torch.Tensor = get_score(
                    model_dir,
                    model_path,
                    input_noise,
                    model_type,
                    phase="train",
                    challenge_name="challenge_with_id.csv",
                    batch_size=batch_size,
                    parallel_batch=num_noise_per_time_step,
                    addt_value=addt_value,
                    t_value=t_value,
                )
                t_value_count += 1
                current_input.append(predictions)
        predictions = torch.cat(current_input, dim=-1)

        predictions = regression_model(predictions).detach().cpu().numpy()
        # clip to [0, 1]
        min_output, max_output = np.min(predictions), np.max(predictions)
        predictions = (predictions - min_output) / (max_output - min_output)
        predictions = torch.tensor(predictions)

        assert torch.all((predictions >= 0) & (predictions <= 1))
        with open(os.path.join(model_dir, predictions_file_name), mode="w", newline="") as file:
            writer = csv.writer(file)
            # Write each value in a separate row
            for value in list(predictions.numpy().squeeze()):
                writer.writerow([value])

    mia_performance_train = evaluate_attack_performance(
        train_indices, "train", tabddpm_data_dir, model_type, predictions_file_name
    )
    mia_performance_val = evaluate_attack_performance(
        val_indices, "test", tabddpm_data_dir, model_type, predictions_file_name
    )
    mia_performance_test = evaluate_attack_performance(
        test_indices, "final", tabddpm_data_dir, model_type, predictions_file_name
    )
    print("MIA performance for training set:")
    print(mia_performance_train)
    print("MIA performance for test set:")
    print(mia_performance_val)
    print("MIA performance for final set:")
    print(mia_performance_test)

    return mia_performance_train, mia_performance_val, mia_performance_test
