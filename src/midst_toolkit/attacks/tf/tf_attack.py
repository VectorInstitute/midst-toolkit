# ruff: noqa: D102, D105, D103, D200, PLR0915
# mypy: disable-error-code=no-untyped-def
# mypy: disable-error-code=has-type
# mypy: disable-error-code=index
# mypy: disable-error-code=attr-defined
# mypy: disable-error-code=assignment
from __future__ import annotations

import csv
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from midst_toolkit.attacks.tf.classification import MLP, fitmodel
from midst_toolkit.attacks.tf.data_utils import (
    CustomUnpickler,
    TaskType,
    evaluate_attack_performance,
    load_multi_table_customized,
    prepare_data_for_attack,
    prepare_fast_dataloader,
)
from midst_toolkit.models.clavaddpm.dataset import Dataset
from midst_toolkit.models.clavaddpm.dataset_transformations import (
    TargetInfo,
)
from midst_toolkit.models.clavaddpm.train import Transformations


def mixed_loss(
    diffusion,
    x,
    out_dict,
    noise=None,
    t=None,
    pt=None,
    return_random=False,
    no_mean=False,
    parallel_batch=None,
    addt_value=None,
):
    device = x.device
    x_num = x[:, : diffusion.num_numerical_features]
    x_cat = x[:, diffusion.num_numerical_features :]

    noise_tensor = torch.tensor(noise, device=device, dtype=torch.float)
    batch_noise = noise_tensor.repeat(x_num.shape[0], 1)

    # there is actually no categorical classes, as we have examined the DM, so we just ignore x_cat here and later
    x_num = x_num.repeat_interleave(parallel_batch, dim=0)
    x_cat = x_cat.repeat_interleave(parallel_batch, dim=0)

    b = x_num.shape[0]

    device = x.device
    if t is None:
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


def build_target(y, policy, task_type):
    info = TargetInfo(policy=policy, mean=None, std=None)
    # info: dict[str, Any] = {"policy": policy}
    if policy is None:
        pass
    elif policy == "default":
        if task_type == TaskType.REGRESSION:
            mean, std = float(y["train"].mean()), float(y["train"].std())
            y = {k: (v - mean) / std for k, v in y.items()}
            info.mean = mean
            info.mean = std
    else:
        raise_unknown("policy", policy)
    return y, info


def transform_dataset(
    dataset: Dataset,
    transformations: Transformations,
    cat_transform=None,
    num_transform=None,
) -> Dataset:
    """
    Applies specified numerical and categorical transformations to a dataset.
    The function processes numerical and categorical features according to the provided
    transformation objects and policies. The order of transformations is important and may
    affect the results.

    Args:
        dataset (Dataset): The input dataset containing numerical and categorical features.
        transformations (Transformations): An object specifying transformation policies for
            categorical and numerical features, including handling of missing values and rare categories.
        cat_transform: Transformer for categorical features. If None, uses the dataset's
            categorical_transform attribute.
        num_transform: Transformer for numerical features. If None, uses the dataset's
            numerical_transform attribute.

    Returns:
        Dataset: A new dataset with transformed numerical and categorical features, and updated target.

    Raises:
        ValueError: If categorical transformation is required but not provided.

    Notes:
        - The order of transformations matters and may affect the output.
        - If the dataset has no categorical features, categorical transformation policies are ignored.
    """
    # WARNING: the order of transformations matters. Moreover, the current
    # implementation is not ideal in that sense.

    if dataset.numerical_features is not None:
        x_num = dataset.numerical_features
        x_num = {k: num_transform.transform(v) for k, v in x_num.items()}

    else:
        raise NotImplementedError("Numerical features are required.")
    if dataset.categorical_features is None:
        assert transformations.categorical_nan_policy is None
        assert transformations.category_minimum_frequency is None
        # assert transformations.cat_encoding is None
        x_cat = None
    else:
        raise NotImplementedError("Categorical features transformation is not implemented yet.")
        # x_cat = process_nans_in_categorical_features(
        #     x_cat,
        #     transformations.categorical_nan_policy,
        # )
        # if transformations.category_minimum_frequency is not None:
        #     x_cat = collapse_rare_categories(
        #         x_cat,
        #         transformations.category_minimum_frequency,
        #     )

        # x_cat = {k: cat_transform.transform(v).astype("float32") for k, v in x_cat.items()}
        # x_num = x_cat if x_num is None else {x: np.hstack([x_num[x], x_cat[x]]) for x in x_num}
        # x_cat = None

    if transformations.target_policy is None:
        y = dataset.target
        y_info = dataset.target_info
    else:
        y, y_info = build_target(dataset.target, transformations.target_policy.value, dataset.task_type.value)
    dataset = replace(dataset, numerical_features=x_num, categorical_features=x_cat, target=y, target_info=y_info)
    dataset.numerical_transform = num_transform
    dataset.categorical_transform = cat_transform

    return dataset


def make_dataset_from_df_with_loaded(
    df, transformation, is_y_cond, df_info=None, std=0, label_encoders=None, num_transform=None
):
    cat_column_orders = []
    num_column_orders = []
    index_to_column = list(df.columns)
    column_to_index = {col: i for i, col in enumerate(index_to_column)}

    if df_info.n_classes == 0:
        x_cat: dict[str, Any] | None = {} if df_info.categorical_column_names is not None else None
        x_num: dict[str, Any] | None = (
            {} if df_info.numerical_column_names is not None or is_y_cond == "concat" else None
        )
        y = {}

        num_cols_with_y = []
        if df_info.numerical_column_names is not None:
            num_cols_with_y += df_info.numerical_column_names
        if is_y_cond == "concat":
            num_cols_with_y = [df_info.target_column_name] + num_cols_with_y

        if len(num_cols_with_y) > 0:
            x_num["train"] = df[num_cols_with_y].values.astype(np.float32)

        y["train"] = df[df_info.target_column_name].values.astype(np.float32)

        if df_info.categorical_column_names is not None:
            x_cat["train"] = df[df_info.categorical_column_names].to_numpy(dtype=np.str_)

        cat_column_orders = [column_to_index[col] for col in df_info.categorical_column_names]
        num_column_orders = [column_to_index[col] for col in num_cols_with_y]
    else:
        raise NotImplementedError("Multitable with classification not supported yet")

    column_orders = num_column_orders + cat_column_orders
    column_orders = [index_to_column[index] for index in column_orders]

    if x_cat is not None and len(df_info.categorical_column_names) > 0:
        x_cat_all = x_cat["train"]
        x_cat_converted = []
        for col_index in range(x_cat_all.shape[1]):
            if label_encoders is None:
                raise ValueError("Should be loaded: label_encoder")
            pass

            x_cat_converted.append(label_encoders[col_index].transform(x_cat_all[:, col_index]).astype(float))

            if std > 0:
                # add noise
                x_cat_converted[-1] += np.random.normal(0, std, x_cat_converted[-1].shape)

        x_cat_converted = np.vstack(x_cat_converted).T

        train_num = x_cat["train"].shape[0]

        x_cat["train"] = np.array(x_cat_converted)[:train_num, :]

        if x_num is not None and len(x_num) > 0:
            x_num["train"] = np.concatenate((x_num["train"], x_cat["train"]), axis=1)
        else:
            x_num = x_cat
            x_cat = None

        target_info = TargetInfo(policy=None, mean=None, std=None)
        dataset = Dataset(
            numerical_features=x_num,
            categorical_features=None,
            target=y,
            target_info=target_info,
            task_type=df_info.task_type,
            n_classes=df_info.n_classes,
            categorical_transform=None,
            numerical_transform=num_transform,
        )
    return (
        transform_dataset(dataset, transformation, None, num_transform=num_transform),
        label_encoders,
        column_orders,
    )


def get_dataset(data_path, target_model_dir=None, train_name="train_with_id.csv", batch_size=None, meta_dir=""):
    tables, relation_order, _ = load_multi_table_customized(
        data_path,
        meta_dir=meta_dir,
        train_name=train_name,
    )
    train_loader_list = []
    if len(relation_order) == 1:
        parent, child = relation_order[0]

        df_with_id = tables[child]["df"]
        df_without_id = df_with_id.drop(columns=[col for col in df_with_id.columns if "_id" in col])

        file_path = os.path.join(target_model_dir, f"{parent}_{child}_ckpt.pkl")

        with open(file_path, "rb") as f:
            model = CustomUnpickler(f).load()

        df_without_id["placeholder"] = df_without_id.index

        # if phase=="train":
        # num_transform = model.dataset.numerical_transform
        # elif phase in ("dev", "final"):
        #         num_transform = model.inverse_transform.numerical_transform
        # else:
        #     raise ValueError("Unknown Phase!!!")
        transformations = model.transformations
        num_transform = model.dataset.numerical_transform
        dataset, _label_encoders, _column_orders = make_dataset_from_df_with_loaded(
            df_without_id,
            transformations,
            is_y_cond=model.model_parameters.is_target_conditioned,
            # data_split_percentages=model.diffusion_config.data_split_ratios,
            df_info=model.table_metadata,
            std=0,
            num_transform=num_transform,
            label_encoders=model.label_encoders,
        )
        dataset.numerical_features["test"] = dataset.numerical_features["train"]
        if dataset.categorical_features is not None:
            dataset.categorical_features["test"] = dataset.categorical_features["train"]
        dataset.target["test"] = dataset.target["train"]
        train_loader = prepare_fast_dataloader(dataset, split="test", batch_size=batch_size, y_type="long")
        train_loader_list.append([train_loader, dataset.numerical_features["test"].shape[0], dataset])

        return train_loader_list
    raise NotImplementedError("Multitable with more than one relation not supported yet")


def get_score(
    data_path: Path,
    save_dir: Path,
    input_noise: list[float],
    model_type: str,
    meta_dir: Path,
    challenge_name: str,
    batch_size: int,
    parallel_batch: int,
    addt_value: int,
    t_value: int,
) -> torch.Tensor:
    """
    Computes the score for a given dataset using a diffusion model.

    Args:
        data_path (Path): Path to the dataset.
        save_dir (Path): Directory where model checkpoints are saved.
        input_noise (list[float]): List of noise values to be used in the loss computation.
        model_type (str): Type of model to use (e.g., "tabddpm").
        meta_dir (Path): Path to the metadata directory.
        challenge_name (str): Name of the challenge dataset.
        batch_size (int): Batch size for data loading.
        parallel_batch (int): Number of parallel batches for processing.
        addt_value (int): Additional value to be passed to the loss function.
        t_value (int): Value of the time step `t` to be used in the loss computation.

    Returns:
        torch.Tensor: A tensor containing the computed loss values.

    Raises:
        ValueError: If the specified `model_type` is not supported.
        AssertionError: If required model checkpoint files are not found or if `iter_max` is not equal to 1.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "tabddpm":
        relation_order = [("None", "trans")]
    elif model_type == "tabsyn":
        raise ValueError("Haven't done it yet!")

    train_loader_list = get_dataset(
        data_path, save_dir, train_name=challenge_name, batch_size=batch_size, meta_dir=meta_dir
    )

    # for tabddpm, relation order only contains like None_trans
    loader_count = 0

    for parent, child in relation_order:
        assert os.path.exists(os.path.join(save_dir, f"{parent}_{child}_ckpt.pkl"))
        train_loader, iter_max, _challenge_dataset = train_loader_list[loader_count]

        filepath = os.path.join(save_dir, f"{parent}_{child}_ckpt.pkl")

        # get the diffusion model
        with open(filepath, "rb") as f:
            model = CustomUnpickler(f).load()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        diffusion = model.diffusion.to(device)

        iter_max = iter_max // batch_size
        assert iter_max == 1

        x, out_dict = next(train_loader)
        out_dict = {"y": out_dict}
        x = x.to(device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(device)

        with torch.no_grad():
            # get loss here
            _noise, t_cur, _pt = mixed_loss(
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
                parallel_batch=parallel_batch,
                addt_value=addt_value,
            )

    return loss


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
    num_noise_per_time_step,
    timesteps_list,
    addt_value_list,
    classifier_num_epochs,
    classifier_learning_rate: float,
    results_path,
    meta_dir: Path,
):
    df_train_merge, _, _ = prepare_data_for_attack(
        indices=train_indices,
        model_type=model_type,
        models_base_dir=Path("/projects/midst-experiments/tabddpm_midst_toolkit/train/"),
        keys_for_deduplication=["trans_id", "balance"],
    )

    df_test_merge, _, _ = prepare_data_for_attack(
        indices=val_indices,
        model_type=model_type,
        models_base_dir=Path("/projects/aieng/midst_competition/data/tabddpm"),
        keys_for_deduplication=["trans_id", "balance"],
    )

    n_feutures = [col for col in df_train_merge.columns if "_id" not in col]
    noise_dimension = len(n_feutures)
    input_noise = [np.random.normal(size=noise_dimension).tolist() for _ in range(num_noise_per_time_step)]

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
            for addt_value in addt_value_list:
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
                        meta_dir=meta_dir,
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
    return input_noise, fitmodel(
        regression_model=regression_model,
        x_train=x_train,
        x_train_label=x_train_label,
        x_val=x_val,
        x_val_label=x_val_label,
        num_epochs=classifier_num_epochs,
        best_model_checkpoint_dir=results_path,
        learning_rate=classifier_learning_rate,
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
    results_path: str,
    test_indices: list[int],
    predictions_file_format: str,
    base_path: Path,
    classifier_num_epochs: int,
    classifier_hidden_dim: int,
    addt_value_list: list[int],
    meta_dir: Path,
    classifier_learning_rate: float,
) -> tuple[Any, Any, Any]:
    os.makedirs(results_path, exist_ok=True)

    input_noise, regression_model = tf_attack_train_classifier(
        train_indices=train_indices,
        val_indices=val_indices,
        samples_per_train_model=samples_per_train_model,
        sample_per_val_model=sample_per_val_model,
        model_type=model_type,
        tabddpm_data_dir=tabddpm_data_dir,
        base_path=base_path,
        target_model_subdir=target_model_subdir,
        classifier_hidden_dim=classifier_hidden_dim,
        num_noise_per_time_step=num_noise_per_time_step,
        timesteps_list=timesteps_list,
        addt_value_list=addt_value_list,
        classifier_num_epochs=classifier_num_epochs,
        results_path=results_path,
        meta_dir=meta_dir,
        classifier_learning_rate=classifier_learning_rate,
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
                    meta_dir=meta_dir,
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
