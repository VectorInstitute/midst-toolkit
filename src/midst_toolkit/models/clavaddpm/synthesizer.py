import os
import pickle
import random
import time
from pathlib import Path
from typing import Any, Literal

import faiss
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder
from torch.nn import functional
from tqdm import tqdm

from midst_toolkit.common.enumerations import DataSplit
from midst_toolkit.models.clavaddpm.dataset import Dataset, Transformations
from midst_toolkit.models.clavaddpm.enumerations import (
    CategoricalEncoding,
    Configs,
    GroupLengthsProbDicts,
    IsTargetCondioned,
    RelationOrder,
    Tables,
)
from midst_toolkit.models.clavaddpm.gaussian_multinomial_diffusion import (
    GaussianMultinomialDiffusion,
)
from midst_toolkit.models.clavaddpm.model import Classifier, ModelParameters
from midst_toolkit.models.clavaddpm.train import get_df_without_id


def sample_from_diffusion(
    df: pd.DataFrame,
    df_info: dict[str, Any],
    diffusion: GaussianMultinomialDiffusion,
    dataset: Dataset,
    label_encoders: dict[int, LabelEncoder],
    sample_size: int,
    model_params: ModelParameters,
    transformations: Transformations,
    sample_batch_size: int = 8192,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Samples synthetic data from a trained diffusion model and aligns it with the real data format.

    Args:
        df: Real data dataframe without id.
        df_info: Dictionary of the real data table information.
        diffusion: The trained diffusion model used for sampling.
        dataset: The dataset object containing training data and transformations.
        label_encoders: The label encoders used to encode the categorical features.
        sample_size: The number of samples to generate.
        model_params: Model parameters including input dimensions and conditioning settings.
        transformations: The transformations used to preprocess the data.
        sample_batch_size: Batch size used in sampling. Defaults to 8192.

    Returns:
        A tuple containing:
            - df_real_data: DataFrame of the real data.
            - df_generated_data: DataFrame of the generated synthetic data.

    """
    num_numerical_features = 0
    if dataset.x_num is not None:
        num_numerical_features = dataset.x_num[DataSplit.TRAIN.value].shape[1]

    category_sizes = dataset.get_category_sizes(DataSplit.TRAIN)
    if len(category_sizes) == 0 or transformations.categorical_encoding == CategoricalEncoding.ONE_HOT:
        category_sizes = [0]

    input_dimension = np.sum(category_sizes) + num_numerical_features
    model_params.d_in = input_dimension

    _, empirical_class_dist = torch.unique(torch.from_numpy(dataset.y[DataSplit.TRAIN.value]), return_counts=True)
    generated_data = diffusion.sample_all(
        sample_size,
        sample_batch_size,
        empirical_class_dist.float(),
        ddim=False,
    )

    generated_features, generated_target = generated_data[0].numpy(), generated_data[1].numpy()

    real_numerical_features = df[df_info["num_cols"]].to_numpy().astype(float)
    real_categorical_features = df[df_info["cat_cols"]].to_numpy().astype(str)
    real_target = np.round(df[df_info["y_col"]].to_numpy().astype(float)).astype(int).reshape(-1, 1)

    if num_numerical_features == 0:
        numerical_features = np.array([])
        categorical_features = np.array([])
    else:
        numerical_features = _get_numerical_features_from_generated_features(
            generated_features,
            num_numerical_features,
            dataset,
            model_params.is_target_conditioned,
            label_encoders,
        )
        categorical_features = _decode_categorical_features(numerical_features, label_encoders)
        numerical_features = _round_numerical_features(numerical_features, real_numerical_features)

    if num_numerical_features != 0 and model_params.is_target_conditioned == IsTargetCondioned.CONCAT:
        generated_target = numerical_features[:, 0]
        numerical_features = numerical_features[:, 1:]

    generated_target = generated_target.reshape(-1, 1)

    if real_categorical_features is not None:
        real_data = np.concatenate((real_numerical_features, real_categorical_features, real_target), axis=1)
        round_target = np.round(generated_target).astype(int)
        generated_data = np.concatenate((numerical_features, categorical_features, round_target), axis=1)
    else:
        real_data = np.concatenate((real_numerical_features, real_target), axis=1)
        generated_data = np.concatenate((numerical_features, np.round(generated_target).astype(int)), axis=1)

    df_real_data = pd.DataFrame(real_data)
    df_generated_data = pd.DataFrame(generated_data)
    columns = [str(x) for x in list(df_real_data.columns)]

    df_real_data.columns = columns
    df_generated_data.columns = columns

    for column in df_real_data.columns:
        if int(column) < real_numerical_features.shape[1]:
            df_real_data[column] = df_real_data[column].astype(float)
            df_generated_data[column] = df_generated_data[column].astype(float)
        elif (
            real_categorical_features is not None
            and int(column) < real_numerical_features.shape[1] + real_categorical_features.shape[1]
        ):
            df_real_data[column] = df_real_data[column].astype(str)
            df_generated_data[column] = df_generated_data[column].astype(str)
        else:
            df_real_data[column] = df_real_data[column].astype(float)
            df_generated_data[column] = df_generated_data[column].astype(float)

    return df_real_data, df_generated_data


def _get_numerical_features_from_generated_features(
    generated_features: np.ndarray,
    num_numerical_features: int,
    dataset: Dataset,
    is_target_conditioned: IsTargetCondioned,
    label_encoders: dict[int, LabelEncoder],
) -> np.ndarray:
    """
    Produce the numerical features from the generated features.

    Args:
        generated_features: The generated features.
        num_numerical_features: The number of numerical features in the real data.
        dataset: The dataset object containing the numerical transformations.
        is_target_conditioned: The condition on the y column.
        label_encoders: The label encoders used to encode the categorical features.

    Returns:
        The numerical features.
    """
    # Checking if it's a regression task and if it's target conditioned.
    # In case either of those are false, we need to add 1 to the number of numerical features to represent the target.
    if dataset.is_regression and is_target_conditioned == IsTargetCondioned.NONE:
        num_numerical_features_sample = num_numerical_features
    else:
        num_numerical_features_sample = num_numerical_features + 1

    assert dataset.numerical_transform is not None
    numerical_features = dataset.numerical_transform.inverse_transform(
        generated_features[:, :num_numerical_features_sample]
    )

    actual_num_numerical_features = num_numerical_features - len(label_encoders)
    return numerical_features[:, :actual_num_numerical_features]


def _decode_categorical_features(
    numerical_features: np.ndarray,
    label_encoders: dict[int, LabelEncoder],
) -> np.ndarray:
    """
    Decode the categorical features from the numerical features using the given label encoders.

    Args:
        numerical_features: The numerical features containing the encoded categorical features.
        label_encoders: The label encoders used to encode the categorical features.

    Returns:
        The categorical features.
    """
    if len(label_encoders) > 0:
        categorical_features = numerical_features[:]  # making a shallow copy of numerical_features
        categorical_features = np.round(categorical_features).astype(int)
        decoded_categorical_features = []
        for column in range(categorical_features.shape[1]):
            categorical_feature = categorical_features[:, column]
            categorical_feature = np.clip(categorical_feature, 0, len(label_encoders[column].classes_) - 1)
            decoded_categorical_features.append(label_encoders[column].inverse_transform(categorical_feature))

        categorical_features = np.column_stack(decoded_categorical_features)
    else:
        categorical_features = np.empty((numerical_features.shape[0], 0))

    return categorical_features


def _round_numerical_features(numerical_features: np.ndarray, real_numerical_features: np.ndarray) -> np.ndarray:
    """
    Round the numerical features to the nearest unique values found in the
    corresponding columns of the real data.

    Args:
        numerical_features: The numerical features to round.
        real_numerical_features: The real numerical features.

    Returns:
        The rounded numerical features.
    """
    discrete_columns = []
    for column in range(real_numerical_features.shape[1]):
        unique_values = np.unique(real_numerical_features[:, column])
        if len(unique_values) <= 32 and ((unique_values - np.round(unique_values)) == 0).all():
            discrete_columns.append(column)

    if discrete_columns:
        numerical_features = round_columns(real_numerical_features, numerical_features, discrete_columns)

    return numerical_features


# TODO: Too many statements and branches, refactor.
def conditional_sampling_by_group_size(  # noqa: PLR0915, PLR0912
    df: pd.DataFrame,
    df_info: dict[str, Any],
    dataset: Dataset,
    label_encoders: dict[int, LabelEncoder],
    classifier: Classifier,
    diffusion: GaussianMultinomialDiffusion,
    group_labels: list[int],
    sample_batch_size: int,
    group_lengths_prob_dicts: dict[int, dict[int, float]],
    is_y_cond: Literal["concat", "embedding", "none"] | None = None,
    classifier_scale: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """
    Samples synthetic data conditionally based on group labels and aligns it with the real data format.

    Args:
        df: Real data dataframe without id.
        df_info: Information about the real data table.
        dataset: The dataset object containing training data and transformations.
        label_encoders: Label encoders for categorical features.
        classifier: The trained classifier model.
        diffusion: The trained diffusion model used for sampling.
        group_labels: List of group labels for conditional sampling.
        sample_batch_size: Batch size used in sampling.
        group_lengths_prob_dicts: Dictionary of group length probabilities for each group label.
        is_y_cond: Conditioning method for the target variable. Can be "concat", "embedding", or "none".
        classifier_scale: Scale factor for the classifier. Defaults to 1.0.

    Returns:
        _description_
    """

    def cond_fn(
        features: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert "y" in kwargs and kwargs["y"] is not None, "The kwargs parameter `y` must be provided."
        assert isinstance(kwargs["y"], torch.Tensor), "The kwargs parameter `y` must be a Tensor."

        y = kwargs["y"]
        remove_first_col = kwargs.get("remove_first_col", False)

        with torch.enable_grad():
            if remove_first_col:
                x_in = features[:, 1:].detach().requires_grad_(True).float()
            else:
                x_in = features.detach().requires_grad_(True).float()
            logits = classifier(x_in, timestep)
            log_probs = functional.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale

    sampled_group_sizes = []
    ys = []
    for group_label in group_labels:
        if group_label not in group_lengths_prob_dicts:
            sampled_group_sizes.append(0)
            continue
        sampled_group_size = sample_from_dict(group_lengths_prob_dicts[group_label])
        assert sampled_group_size is not None
        sampled_group_sizes.append(sampled_group_size)
        ys.extend([group_label] * sampled_group_size)

    all_rows = []
    all_clusters = []
    curr_index = 0
    while curr_index < len(ys):
        end_index = min(curr_index + sample_batch_size, len(ys))
        curr_ys = torch.tensor(np.array(ys[curr_index:end_index]).reshape(-1, 1), requires_grad=False)
        curr_model_kwargs = {}
        curr_model_kwargs["y"] = curr_ys
        curr_sample, _ = diffusion.conditional_sample(targets=curr_ys, model_kwargs=curr_model_kwargs, cond_fn=cond_fn)
        all_rows.extend([sample.cpu().numpy() for sample in [curr_sample]])
        all_clusters.extend([curr_ys.cpu().numpy() for curr_ys in [curr_ys]])
        curr_index += sample_batch_size

    arr = np.concatenate(all_rows, axis=0)
    cluster_arr = np.concatenate(all_clusters, axis=0)

    num_numerical_features = dataset.x_num["train"].shape[1] if dataset.x_num is not None else 0

    x_gen, y_gen = arr, cluster_arr
    num_numerical_features_sample = num_numerical_features + int(dataset.is_regression and not is_y_cond)

    x_num_real = df[df_info["num_cols"]].to_numpy().astype(float)
    x_cat_real = df[df_info["cat_cols"]].to_numpy().astype(str)
    y_real = np.round(df[df_info["y_col"]].to_numpy().astype(float)).astype(int).reshape(-1, 1)

    x_num_ = x_gen

    if num_numerical_features != 0:
        assert dataset.numerical_transform is not None
        x_num_ = dataset.numerical_transform.inverse_transform(x_gen[:, :num_numerical_features_sample])
        actual_num_numerical_features = num_numerical_features - len(label_encoders)
        x_num = x_num_[:, :actual_num_numerical_features]
        if len(label_encoders) > 0:
            x_cat = x_num_[:, actual_num_numerical_features:]
            x_cat = np.round(x_cat).astype(int)
            decoded_x_cat = []
            for col in range(x_cat.shape[1]):
                decoded_x_cat.append(label_encoders[col].inverse_transform(x_cat[:, col]))
            x_cat = np.column_stack(decoded_x_cat)

        disc_cols = []
        for col in range(x_num_real.shape[1]):
            uniq_vals = np.unique(x_num_real[:, col])
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)
        # print("Discrete cols:", disc_cols)
        if is_y_cond == "concat":
            y_gen = x_num[:, 0]
            x_num = x_num[:, 1:]
        if disc_cols:
            x_num = round_columns(x_num_real, x_num, disc_cols)

    y_gen = y_gen.reshape(-1, 1)

    if x_cat_real is not None and x_cat_real.shape[1] > 0:
        total_real = np.concatenate((x_num_real, x_cat_real, y_real), axis=1)
        gen_real = np.concatenate((x_num, x_cat, np.round(y_gen).astype(int)), axis=1)

    else:
        total_real = np.concatenate((x_num_real, y_real), axis=1)
        gen_real = np.concatenate((x_num, np.round(y_gen).astype(int)), axis=1)

    df_total = pd.DataFrame(total_real)
    df_gen = pd.DataFrame(gen_real)
    columns = [str(x) for x in list(df_total.columns)]

    df_total.columns = columns
    df_gen.columns = columns

    for column in df_total.columns:
        col_str: str = str(column)
        col_int: int = int(column)
        if col_int < x_num_real.shape[1]:
            df_total[col_str] = df_total[col_str].astype(float)
            df_gen[col_str] = df_gen[col_str].astype(float)
        elif x_cat_real is not None and col_int < x_num_real.shape[1] + x_cat_real.shape[1]:
            df_total[col_str] = df_total[col_str].astype(str)
            df_gen[col_str] = df_gen[col_str].astype(str)
        else:
            df_total[col_str] = df_total[col_str].astype(float)
            df_gen[col_str] = df_gen[col_str].astype(float)

    return df_total, df_gen, sampled_group_sizes


def handle_multi_parent(
    child: str,
    parents: list[str],
    synthetic_tables: dict[tuple[str, str], dict[str, Any]],
    n_clusters: int,
    unique_matching: bool = True,
    batch_size: int = 100,
    no_matching: bool = False,
) -> pd.DataFrame:
    """
    Handles the matching process for a child table with multiple parent tables.

    Args:
        child: Name of the child table.
        parents: List of parent table names.
        synthetic_tables: Dictionary containing synthetic tables with dataframes
            and ``(parent, child)`` keys.
        n_clusters: Number of clusters to use in the matching process.
        unique_matching: Whether to enforce unique matching. Defaults to True.
        batch_size: Batch size used in the matching process. Defaults to 100.
        no_matching: Whether to skip the matching process and randomly shuffle. Defaults to False.

    Returns:
        DataFrame of the matched child table.
    """
    synthetic_child_dfs = [(synthetic_tables[(parent, child)]["df"].copy(), parent) for parent in parents]
    anchor_index = np.argmin([len(df) for df, _ in synthetic_child_dfs])
    anchor = synthetic_child_dfs[anchor_index]
    synthetic_child_dfs.pop(anchor_index)
    for df, parent in synthetic_child_dfs:
        df_without_ids = get_df_without_id(df)
        anchor_df_without_ids = get_df_without_id(anchor[0])
        df_val = df_without_ids.values.astype(float)
        anchor_val = anchor_df_without_ids.values.astype(float)
        if len(df_val.shape) == 1:
            df_val = df_val.reshape(-1, 1)
            anchor_val = anchor_val.reshape(-1, 1)

        indices, _ = match_tables(
            anchor_val,
            df_val,
            n_clusters=n_clusters,
            unique_matching=unique_matching,
            batch_size=batch_size,
        )
        if no_matching:
            # randomly shuffle the array
            indices = np.random.permutation(indices).tolist()

        df = df.iloc[indices]
        anchor[0][f"{parent}_id"] = df[f"{parent}_id"].values
    return anchor[0]


def match_tables(
    a_numpy_array: np.ndarray,
    b_numpy_array: np.ndarray,
    n_clusters: int = 25,
    unique_matching: bool = True,
    batch_size: int = 100,
) -> tuple[list[int], list[float]]:
    """
    Matches rows from two tables A and B using FAISS for efficient nearest neighbor search.

    Args:
        a_numpy_array: Numpy array representing table A.
        b_numpy_array: Numpy array representing table B.
        n_clusters: Number of clusters to use in the matching process. Defaults to 25.
        unique_matching: Whether to enforce unique matching. Defaults to True.
        batch_size: Batch size used in the matching process. Defaults to 100.

    Returns:
        A tuple containing:
            - indices: List of indices in b_numpy_array that match each row in a_numpy_array.
            - distances: List of distances corresponding to the matches.
    """
    a_array = np.ascontiguousarray(a_numpy_array, dtype=np.float32)
    b_array = np.ascontiguousarray(b_numpy_array, dtype=np.float32)

    # Dimension of vectors
    b_array_dimension = b_array.shape[1]

    if unique_matching:
        quantiser = faiss.IndexFlatL2(b_array_dimension)
        index = faiss.IndexIVFFlat(quantiser, b_array_dimension, n_clusters, faiss.METRIC_L2)
    else:
        res = faiss.StandardGpuResources()
        quantiser = faiss.IndexFlatL2(b_array_dimension)
        index_cpu = faiss.IndexIVFFlat(quantiser, b_array_dimension, n_clusters, faiss.METRIC_L2)
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)

    index.train(b_array)
    index.add(b_array)

    # Initialize lists to store the results
    all_indices = []
    all_distances = []

    if unique_matching:
        batch_size = 1
        n_batches = (a_array.shape[0] + batch_size - 1) // batch_size

        for i in tqdm(range(n_batches)):
            start = i * batch_size
            end = min((i + 1) * batch_size, a_array.shape[0])
            distance, search_indices = index.search(a_array[start:end], k=1)
            index.remove_ids(search_indices.flatten())
            all_distances.append(distance)
            all_indices.append(search_indices)

        # Concatenate the results from all batches
        all_distances_np = np.vstack(all_distances)
        all_indices_np = np.vstack(all_indices)
        distances = all_distances_np.flatten().tolist()
        indices = all_indices_np.flatten().tolist()
    else:
        n_batches = (a_array.shape[0] + batch_size - 1) // batch_size

        for i in tqdm(range(n_batches)):
            start = i * batch_size
            end = min((i + 1) * batch_size, a_array.shape[0])
            distance, search_indices = index.search(a_array[start:end], k=1)
            all_distances.append(distance)
            all_indices.append(search_indices)

        # Concatenate the results from all batches
        all_distances_np = np.vstack(all_distances)
        all_indices_np = np.vstack(all_indices)
        distances = all_distances_np.flatten().tolist()
        indices = all_indices_np.flatten().tolist()
        indices = convert_to_unique_indices(indices)
        assert len(indices) == len(set(indices))

    return indices, distances


def round_columns(
    real_features: np.ndarray,
    synthetic_features: np.ndarray,
    columns: list[int],
) -> np.ndarray:
    """
    Rounds the values in specified columns of the synthetic data to the nearest
    unique values found in the corresponding columns of the real data.

    Args:
        real_features: Numpy array representing the real data.
        synthetic_features: Numpy array representing the synthetic data.
        columns: List of columns to round.

    Returns:
        Numpy array representing the rounded synthetic data.
    """
    for column in columns:
        unique_features = np.unique(real_features[:, column])
        distances = cdist(
            synthetic_features[:, column][:, np.newaxis].astype(float),
            unique_features[:, np.newaxis].astype(float),
        )
        synthetic_features[:, column] = unique_features[distances.argmin(axis=1)]
    return synthetic_features


def sample_from_dict(probabilities: dict[int, float]) -> int:
    """
    Samples an integer key from a dictionary based on the provided probabilities.

    Args:
        probabilities: Dictionary of integer keys and their corresponding probabilities.
            The sum of all probabilities must be 1.0.

    Returns:
        The sampled key.
    """
    assert sum(probabilities.values()) == 1.0, "The sum of all probabilities must be 1.0."

    # Generate a random number between 0 and 1
    random_number = random.random()

    # Initialize cumulative sum and the selected key
    cumulative_sum = 0.0

    # Iterate through the dictionary
    for key, probability in probabilities.items():
        cumulative_sum += probability
        if cumulative_sum >= random_number:
            # return the key if the cumulative sum is greater than or equal to the random number
            return key

    raise Exception("Unable to sample from dictionary.")


def convert_to_unique_indices(indices: list[int]) -> list[int]:
    """
    Converts a list of indices to ensure all indices are unique by replacing duplicates
    with the smallest available integers not already in the list.

    Args:
        indices: List of indices to convert.

    Returns:
        List of unique indices.
    """
    occurrence = set()
    max_index = len(indices)  # Assuming the range is the length of the list
    replacement_candidates = set(range(max_index)) - set(indices)

    for i, num in enumerate(tqdm(indices)):
        if num in occurrence:
            # Find the smallest number not in the list
            replacement = min(replacement_candidates)
            indices[i] = replacement
            replacement_candidates.remove(replacement)
        else:
            occurrence.add(num)

    return indices


def clava_synthesizing_matching_process(
    synthetic_tables: dict[tuple[str, str], dict[str, Any]],
    tables: Tables,
    relation_order: RelationOrder,
    configs: Configs,
) -> dict[str, pd.DataFrame]:
    """
    Matches synthetic child tables to synthetic parent tables based on clustering information.

    Args:
        synthetic_tables: Dictionary containing synthetic dataframes for each parent-child relationship.
        tables: Original tables containing dataframes and clustering information.
        relation_order: List of parent-child table relationships.
        configs: Configuration with matching settings.

    Returns:
        Dictionary containing the matched synthetic child tables.
    """
    final_tables: dict[str, pd.DataFrame] = {}
    for parent, child in relation_order:
        if child not in final_tables:
            if len(tables[child]["parents"]) > 1:
                final_tables[child] = handle_multi_parent(
                    child,
                    tables[child]["parents"],
                    synthetic_tables,
                    configs["matching"]["num_matching_clusters"],
                    unique_matching=configs["matching"]["unique_matching"],
                    batch_size=configs["matching"]["matching_batch_size"],
                    no_matching=configs["matching"]["no_matching"],
                )
            else:
                final_tables[child] = synthetic_tables[(parent, child)]["df"]
    return final_tables


# TODO: Too many statements and branches, refactor.
def clava_synthesizing(  # noqa: PLR0915, PLR0912
    tables: Tables,
    relation_order: RelationOrder,
    save_dir: Path,
    all_group_lengths_prob_dicts: GroupLengthsProbDicts,
    models: dict[tuple[str, str], dict[str, Any]],
    configs: Configs,
    sample_scale: float = 1.0,
) -> tuple[dict[str, pd.DataFrame], float, float]:
    """
    Synthesizes new data for single-table or multi-table datasets using trained models and
    clustering information.

    Args:
        tables: Tables containing dataframes and clustering information.
        relation_order: List of parent-child table relationships.
        save_dir: Directory to save intermediate and final results.
        all_group_lengths_prob_dicts: Dictionary containing group length probabilities for each
            parent-child relationship.
        models: Trained models for each parent-child relationship.
        configs: Configuration settings for synthesis and matching.
        sample_scale: Scale factor for the number of samples to generate
            based on the train data size. Defaults to 1.0.

    Returns:
        A tuple containing:
            - cleaned_tables: Synthesized tables with original columns.
            - synthesizing_time_spent: Time taken for the synthesis process.
            - matching_time_spent: Time taken for the matching process.
    """
    synthesizing_start_time = time.time()
    synthetic_tables: dict[tuple[str, str], dict[str, Any]] = {}

    # Synthesize
    for parent, child in relation_order:
        print(f"Generating {parent} -> {child}")
        result = models[(parent, child)]
        df_with_cluster = tables[child]["df"]
        df_without_id = get_df_without_id(df_with_cluster)

        print("Sample size: {}".format(int(sample_scale * len(df_without_id))))

        if parent is None:
            _, child_generated = sample_from_diffusion(
                df=df_without_id,
                df_info=result["df_info"],
                diffusion=result["diffusion"],
                dataset=result["dataset"],
                label_encoders=result["label_encoders"],
                sample_size=int(sample_scale * len(df_without_id)),
                model_params=ModelParameters(**result["model_params"]),
                transformations=Transformations(**result["T_dict"]),
                sample_batch_size=configs["sampling"]["batch_size"],
            )
            child_keys = list(range(len(child_generated)))
            generated_final_arr = np.concatenate(
                [np.array(child_keys).reshape(-1, 1), child_generated.to_numpy()],
                axis=1,
            )
            generated_final_df = pd.DataFrame(
                generated_final_arr,
                columns=[f"{child}_id"]
                + result["df_info"]["num_cols"]
                + result["df_info"]["cat_cols"]
                + [result["df_info"]["y_col"]],
            )
            # generated_final_df = generated_final_df[tables[child]['df'].columns]
            generated_final_df = generated_final_df[[f"{child}_id"] + df_without_id.columns.tolist()]
            synthetic_tables[(parent, child)] = {
                "df": generated_final_df,
                "keys": child_keys,
            }
        else:
            for key, val in synthetic_tables.items():
                if key[1] == parent:
                    parent_synthetic_df = val["df"]
                    parent_keys = val["keys"]
                    parent_result = models[key]
                    break

            child_result = models[(parent, child)]
            parent_label_index = parent_result["column_orders"].index(child_result["df_info"]["y_col"])

            parent_synthetic_df_without_id = get_df_without_id(parent_synthetic_df)

            (
                _,
                child_generated,
                child_sampled_group_sizes,
            ) = conditional_sampling_by_group_size(
                df=df_without_id,
                df_info=child_result["df_info"],
                dataset=child_result["dataset"],
                label_encoders=child_result["label_encoders"],
                classifier=child_result["classifier"],
                diffusion=child_result["diffusion"],
                group_labels=parent_synthetic_df_without_id.values[:, parent_label_index]
                .astype(float)
                .astype(int)
                .tolist(),
                group_lengths_prob_dicts=all_group_lengths_prob_dicts[(parent, child)],
                sample_batch_size=configs["sampling"]["batch_size"],
                is_y_cond="none",
                classifier_scale=configs["sampling"]["classifier_scale"],
            )

            child_foreign_keys = np.repeat(parent_keys, child_sampled_group_sizes, axis=0).reshape((-1, 1))
            child_foreign_keys_arr = np.array(child_foreign_keys).reshape(-1, 1)
            child_primary_keys_arr = np.arange(len(child_generated)).reshape(-1, 1)

            child_generated_final_arr = np.concatenate(
                [
                    child_primary_keys_arr,
                    child_generated.to_numpy(),
                    child_foreign_keys_arr,
                ],
                axis=1,
            )

            child_final_columns = (
                [f"{child}_id"]
                + result["df_info"]["num_cols"]
                + result["df_info"]["cat_cols"]
                + [result["df_info"]["y_col"]]
                + [f"{parent}_id"]
            )

            child_final_df = pd.DataFrame(child_generated_final_arr, columns=child_final_columns)
            original_columns = []
            for col in tables[child]["df"].columns:
                if col in child_final_df.columns:
                    original_columns.append(col)
            child_final_df = child_final_df[original_columns]
            synthetic_tables[(parent, child)] = {
                "df": child_final_df,
                "keys": child_primary_keys_arr.flatten().tolist(),
            }
        with open(os.path.join(save_dir, "before_matching/synthetic_tables.pkl"), "wb") as file:
            pickle.dump(synthetic_tables, file)

    synthesizing_end_time = time.time()
    synthesizing_time_spent = synthesizing_end_time - synthesizing_start_time

    matching_start_time = time.time()
    # Matching
    final_tables = clava_synthesizing_matching_process(synthetic_tables, tables, relation_order, configs)
    matching_end_time = time.time()
    matching_time_spent = matching_end_time - matching_start_time

    cleaned_tables: dict[str, pd.DataFrame] = {}
    for table_key, table_val in final_tables.items():
        if "account_id" in tables[table_key]["original_cols"]:
            cols = tables[table_key]["original_cols"]
            cols.remove("account_id")
        else:
            cols = tables[table_key]["original_cols"]
        cleaned_tables[table_key] = pd.DataFrame(table_val[cols])

    for cleaned_key, cleaned_val in cleaned_tables.items():
        table_dir = os.path.join(
            configs["general"]["workspace_dir"],
            configs["general"]["exp_name"],
            cleaned_key,
            f"{configs['general']['sample_prefix']}_final",
        )
        os.makedirs(table_dir, exist_ok=True)
        if f"{cleaned_key}_id" in cleaned_val.columns:
            cleaned_val.to_csv(
                os.path.join(table_dir, f"{cleaned_key}_synthetic_with_id.csv"),
                index=False,
            )
            val_no_id = cleaned_val.drop(columns=[f"{cleaned_key}_id"])
            val_no_id.to_csv(os.path.join(table_dir, f"{cleaned_key}_synthetic.csv"), index=False)
        else:
            cleaned_val.to_csv(os.path.join(table_dir, f"{cleaned_key}_synthetic.csv"), index=False)
    return cleaned_tables, synthesizing_time_spent, matching_time_spent
