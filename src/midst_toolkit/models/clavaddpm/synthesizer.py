import pickle
import random
import time
from logging import INFO
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder
from torch.nn import functional
from tqdm import tqdm

from midst_toolkit.common.enumerations import DataSplit
from midst_toolkit.common.logger import log
from midst_toolkit.models.clavaddpm.dataset import Dataset, Transformations
from midst_toolkit.models.clavaddpm.enumerations import (
    CategoricalEncoding,
    Configs,
    GroupLengthProbDict,
    GroupLengthsProbDicts,
    IsTargetConditioned,
    ModelArtifacts,
    Relation,
    RelationOrder,
    Tables,
)
from midst_toolkit.models.clavaddpm.gaussian_multinomial_diffusion import (
    ConditioningFunction,
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
            - df_synthetic_data: DataFrame of the generated synthetic data.

    """
    num_features = 0
    if dataset.numerical_features is not None:
        num_features = dataset.numerical_features[DataSplit.TRAIN.value].shape[1]

    category_sizes = dataset.get_category_sizes(DataSplit.TRAIN)
    if len(category_sizes) == 0 or transformations.categorical_encoding == CategoricalEncoding.ONE_HOT:
        category_sizes = [0]

    model_params.input_dimension = np.sum(category_sizes) + num_features

    _, empirical_class_dist = torch.unique(torch.from_numpy(dataset.target[DataSplit.TRAIN.value]), return_counts=True)
    synthetic_data = diffusion.sample_all(
        sample_size,
        sample_batch_size,
        empirical_class_dist.float(),
        ddim=False,
    )

    synthetic_features, synthetic_target = synthetic_data[0].numpy(), synthetic_data[1].numpy()

    df_real_data, df_synthetic_data = _post_process_synthetic_data(
        synthetic_features,
        synthetic_target,
        df,
        df_info,
        num_features,
        model_params.is_target_conditioned,
        dataset,
        label_encoders,
    )

    return df_real_data, df_synthetic_data


def conditional_sample_from_diffusion(
    df: pd.DataFrame,
    df_info: dict[str, Any],
    dataset: Dataset,
    label_encoders: dict[int, LabelEncoder],
    classifier: Classifier,
    diffusion: GaussianMultinomialDiffusion,
    group_labels: list[int],
    sample_batch_size: int,
    group_length_prob_dict: GroupLengthProbDict,
    is_target_conditioned: IsTargetConditioned = IsTargetConditioned.NONE,
    classifier_scale: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """
    Samples synthetic data conditionally based on group labels and a trained diffusion model,
    and aligns it with the real data format.

    Args:
        df: Real data dataframe without id.
        df_info: Information about the real data table.
        dataset: The dataset object containing training data and transformations.
        label_encoders: Label encoders for categorical features.
        classifier: The trained classifier model.
        diffusion: The trained diffusion model used for sampling.
        group_labels: List of group labels for conditional sampling.
        sample_batch_size: Batch size used in sampling.
        group_length_prob_dict: Dictionary of group length probabilities for each group label.
        is_target_conditioned: Conditioning method for the target variable. Defaults to IsTargetConditioned.NONE.
        classifier_scale: Scale factor for the classifier. Defaults to 1.0.

    Returns:
        A tuple containing:
            - df_real_data: DataFrame of the real data.
            - df_synthetic_data: DataFrame of the generated synthetic data.
            - sampled_group_sizes: List of the sampled group sizes.
    """
    num_features = 0
    if dataset.numerical_features is not None:
        num_features = dataset.numerical_features[DataSplit.TRAIN.value].shape[1]

    targets, sampled_group_sizes = _sample_targets(group_labels, group_length_prob_dict)

    synthetic_features, synthetic_targets = _get_synthetic_data_by_conditional_sample(
        targets,
        sample_batch_size,
        classifier,
        classifier_scale,
        diffusion,
    )

    df_real_data, df_synthetic_data = _post_process_synthetic_data(
        synthetic_features,
        synthetic_targets,
        df,
        df_info,
        num_features,
        is_target_conditioned,
        dataset,
        label_encoders,
    )

    return df_real_data, df_synthetic_data, sampled_group_sizes


def _post_process_synthetic_data(
    synthetic_features: np.ndarray,
    synthetic_target: np.ndarray,
    df: pd.DataFrame,
    df_info: dict[str, Any],
    num_features: int,
    is_target_conditioned: IsTargetConditioned,
    dataset: Dataset,
    label_encoders: dict[int, LabelEncoder],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Post-processes the synthetic data to align it with the real data format.

    Args:
        synthetic_features: The synthetic features.
        synthetic_target: The synthetic targets.
        df: The dataframe containing the real data.
        df_info: The info dictionary of the real data table.
        num_features: The number of features.
        is_target_conditioned: The condition on the y column.
        dataset: The dataset object containing the numerical transformations.
        label_encoders: The label encoders used to encode the categorical features.

    Returns:
        A tuple containing:
            - df_real_data: DataFrame of the real data.
            - df_synthetic_data: DataFrame of the generated synthetic data.
    """
    real_numerical_features = df[df_info["num_cols"]].to_numpy().astype(float)
    real_categorical_features = df[df_info["cat_cols"]].to_numpy().astype(str)
    real_target = np.round(df[df_info["y_col"]].to_numpy().astype(float)).astype(int).reshape(-1, 1)

    if num_features == 0:
        numerical_features = np.array([])
        categorical_features = np.array([])
    else:
        all_features = _get_all_features_from_synthetic_features(
            synthetic_features,
            dataset,
            is_target_conditioned,
        )
        numerical_features, encoded_categorical_features = _split_features(all_features, label_encoders)

        categorical_features = _decode_categorical_features(encoded_categorical_features, label_encoders)

        if is_target_conditioned == IsTargetConditioned.CONCAT:
            synthetic_target = numerical_features[:, 0]
            numerical_features = numerical_features[:, 1:]

        numerical_features = _round_discrete_numerical_features(numerical_features, real_numerical_features)

    synthetic_target = synthetic_target.reshape(-1, 1)

    if real_categorical_features is not None:
        real_data = np.concatenate((real_numerical_features, real_categorical_features, real_target), axis=1)
        round_target = np.round(synthetic_target).astype(int)
        synthetic_data = np.concatenate((numerical_features, categorical_features, round_target), axis=1)
    else:
        real_data = np.concatenate((real_numerical_features, real_target), axis=1)
        synthetic_data = np.concatenate((numerical_features, np.round(synthetic_target).astype(int)), axis=1)

    df_real_data = pd.DataFrame(real_data)
    df_synthetic_data = pd.DataFrame(synthetic_data)
    columns = [str(x) for x in list(df_real_data.columns)]

    df_real_data.columns = columns
    df_synthetic_data.columns = columns

    for column in df_real_data.columns:
        if int(column) < real_numerical_features.shape[1]:
            df_real_data[column] = df_real_data[column].astype(float)
            df_synthetic_data[column] = df_synthetic_data[column].astype(float)
        elif (
            real_categorical_features is not None
            and int(column) < real_numerical_features.shape[1] + real_categorical_features.shape[1]
        ):
            df_real_data[column] = df_real_data[column].astype(str)
            df_synthetic_data[column] = df_synthetic_data[column].astype(str)
        else:
            df_real_data[column] = df_real_data[column].astype(float)
            df_synthetic_data[column] = df_synthetic_data[column].astype(float)

    return df_real_data, df_synthetic_data


def _get_all_features_from_synthetic_features(
    synthetic_features: np.ndarray,
    dataset: Dataset,
    is_target_conditioned: IsTargetConditioned,
) -> np.ndarray:
    """
    Produce a dataset with all features from the generated synthetic features.

    Args:
        synthetic_features: The generated synthetic features.
        dataset: The dataset object containing the numerical transformations.
        is_target_conditioned: The condition on the y column.

    Returns:
        All features from the synthetic features.
    """
    num_features = synthetic_features.shape[1]

    # In case it's a regression task and it's not target conditioned,
    # we need to add 1 to the number of numerical features to represent the target.
    if dataset.is_regression and is_target_conditioned == IsTargetConditioned.NONE:
        num_features_sample = num_features + 1
    else:
        num_features_sample = num_features

    assert dataset.numerical_transform is not None
    return dataset.numerical_transform.inverse_transform(synthetic_features[:, :num_features_sample])


def _split_features(
    all_features: np.ndarray,
    label_encoders: dict[int, LabelEncoder],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the features into numerical and categorical features.

    Args:
        all_features: The data with all features.
        label_encoders: The label encoders used to encode the categorical features.

    Returns:
        A tuple containing the numerical and categorical features.
    """
    num_numerical_features = all_features.shape[1] - len(label_encoders)
    numerical_features = all_features[:, :num_numerical_features]
    categorical_features = all_features[:, num_numerical_features:]

    return numerical_features, categorical_features


def _decode_categorical_features(
    encoded_categorical_features: np.ndarray,
    label_encoders: dict[int, LabelEncoder],
) -> np.ndarray:
    """
    Decode the encoded categorical features using the given label encoders.

    Args:
        encoded_categorical_features: The encoded categorical features.
        label_encoders: The label encoders used to encode the categorical features.

    Returns:
        The categorical features.
    """
    if len(label_encoders) == 0:
        return np.empty((encoded_categorical_features.shape[0], 0))

    categorical_features = np.round(encoded_categorical_features).astype(int)
    decoded_categorical_features = []
    for column in range(categorical_features.shape[1]):
        categorical_feature = categorical_features[:, column]
        categorical_feature = np.clip(categorical_feature, 0, len(label_encoders[column].classes_) - 1)
        decoded_categorical_features.append(label_encoders[column].inverse_transform(categorical_feature))

    return np.column_stack(decoded_categorical_features)


def _round_discrete_numerical_features(
    numerical_features: np.ndarray,
    real_numerical_features: np.ndarray,
) -> np.ndarray:
    """
    Round the discrete numerical features to the nearest unique values found in the
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


def _sample_targets(
    group_labels: list[int],
    group_length_prob_dict: GroupLengthProbDict,
) -> tuple[list[int], list[int]]:
    """
    Samples targets for the conditional sampling.

    Args:
        group_labels: List of group labels.
        group_length_prob_dict: Dictionary of group length probabilities for each group label.

    Returns:
        A tuple containing:
            - targets: List of targets.
            - sampled_group_sizes: List of sampled group sizes.
    """
    sampled_group_sizes = []
    targets = []
    for group_label in group_labels:
        if group_label not in group_length_prob_dict:
            sampled_group_sizes.append(0)
            continue

        sampled_group_size = sample_from_dict(group_length_prob_dict[group_label])
        sampled_group_sizes.append(sampled_group_size)
        targets.extend([group_label] * sampled_group_size)

    return targets, sampled_group_sizes


def _get_synthetic_data_by_conditional_sample(
    targets: list[int],
    sample_batch_size: int,
    classifier: Classifier,
    classifier_scale: float,
    diffusion: GaussianMultinomialDiffusion,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates features and targets using the classifier and diffusion model with conditional sampling.

    Args:
        targets: List of targets.
        sample_batch_size: Batch size used in sampling.
        classifier: The classifier model.
        classifier_scale: Scale factor for the classifier.
        diffusion: The diffusion model.

    Returns:
        A tuple containing:
            - The synthetic features from the diffusion model's conditional sampling.
            - The synthetic targets from the diffusion model's conditional sampling.
    """
    all_rows = []
    all_clusters = []
    curr_index = 0
    conditioning_function = _get_conditioning_function(classifier, classifier_scale)
    while curr_index < len(targets):
        end_index = min(curr_index + sample_batch_size, len(targets))
        curr_targets = torch.tensor(np.array(targets[curr_index:end_index]).reshape(-1, 1), requires_grad=False)

        curr_sample, _ = diffusion.conditional_sample(
            targets=curr_targets,
            model_kwargs={"target": curr_targets},
            conditioning_function=conditioning_function,
        )

        all_rows.append(curr_sample.cpu().clone().numpy())
        all_clusters.append(curr_targets.cpu().clone().numpy())

        curr_index += sample_batch_size

    return np.concatenate(all_rows, axis=0), np.concatenate(all_clusters, axis=0)


def _get_conditioning_function(classifier: Classifier, classifier_scale: float) -> ConditioningFunction:
    def conditioning_function(
        features: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert "target" in kwargs and kwargs["target"] is not None, "The kwargs parameter `target` must be provided."
        assert isinstance(kwargs["target"], torch.Tensor), "The kwargs parameter `target` must be a Tensor."

        target = kwargs["target"]
        remove_first_col = kwargs.get("remove_first_col", False)

        with torch.enable_grad():
            if remove_first_col:
                x_in = features[:, 1:].detach().requires_grad_(True).float()
            else:
                x_in = features.detach().requires_grad_(True).float()
            logits = classifier(x_in, timestep)
            log_probs = functional.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), target.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale

    return conditioning_function


def handle_multi_parent(
    child: str,
    parents: list[str],
    synthetic_tables: dict[Relation, dict[str, Any]],
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
    synthetic_tables: dict[Relation, dict[str, Any]],
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


def clava_synthesizing(
    tables: Tables,
    relation_order: RelationOrder,
    save_dir: Path,
    all_group_lengths_prob_dicts: GroupLengthsProbDicts,
    models: dict[Relation, ModelArtifacts],
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
            - synthetic_data: Synthesized data with original columns.
            - synthesizing_time_spent: Time taken for the synthesis process.
            - matching_time_spent: Time taken for the matching process.
    """
    synthesizing_start_time = time.time()
    synthetic_tables: dict[Relation, dict[str, Any]] = {}

    # Synthesize
    for parent, child in relation_order:
        log(INFO, f"Generating {parent} -> {child}")
        training_results = models[(parent, child)]
        df_with_cluster = tables[child]["df"]
        df_without_id = get_df_without_id(df_with_cluster)

        log(INFO, "Sample size: {}".format(int(sample_scale * len(df_without_id))))

        if parent is None:
            # synthesize data for single table or tables with no parent
            synthesized_df, table_keys = _synthesize_single_table(
                child,
                df_without_id,
                training_results,
                sample_scale,
                configs["sampling"]["batch_size"],
            )
        else:
            # Finding previously synthesized data and training results for the parent
            parent_synthetic_data = None
            parent_training_results = None
            for key, val in synthetic_tables.items():
                if key[1] == parent:
                    parent_synthetic_data = val
                    parent_training_results = models[key]
                    break

            assert parent_synthetic_data is not None, f"Could not find synthetic data for parent table '{parent}'."
            assert parent_training_results is not None, f"Could not find training results for parent table '{parent}'."

            # Synthesize data for tables with (parent, child) relationship
            synthesized_df, table_keys = _synthesize_multi_table(
                parent,
                child,
                parent_training_results,
                training_results,
                parent_synthetic_data,
                df_without_id,
                all_group_lengths_prob_dicts[(parent, child)],
                tables,
                configs["sampling"]["batch_size"],
                configs["sampling"]["classifier_scale"],
            )

        synthetic_tables[(parent, child)] = {
            "df": synthesized_df,
            "keys": table_keys,
        }

        before_matching_dir = save_dir / "before_matching"
        before_matching_dir.mkdir(parents=True, exist_ok=True)
        with open(before_matching_dir / "synthetic_tables.pkl", "wb") as file:
            pickle.dump(synthetic_tables, file)

    synthesizing_end_time = time.time()
    synthesizing_time_spent = synthesizing_end_time - synthesizing_start_time

    # Matching
    matching_start_time = time.time()

    synthetic_data = clava_synthesizing_matching_process(synthetic_tables, tables, relation_order, configs)

    matching_end_time = time.time()
    matching_time_spent = matching_end_time - matching_start_time

    cleaned_synthetic_data = _clean_and_save_synthetic_data(synthetic_data, tables, configs)
    return cleaned_synthetic_data, synthesizing_time_spent, matching_time_spent


def _synthesize_single_table(
    table_name: str,
    data: pd.DataFrame,
    training_results: ModelArtifacts,
    sample_scale: float,
    sample_batch_size: int,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Synthesizes data for single table using the trained diffusion model.

    Args:
        table_name: Name of the table to synthesize.
        data: DataFrame containing the real data to be used for synthesizing.
        training_results: Dictionary containing the training results, including the trained diffusion model.
        sample_scale: Scale factor for the number of samples to generate. Will be used to determine the
            number of samples to generate by multipling the ``data`` size by ``sample_scale``.
        sample_batch_size: Batch size for sampling.

    Returns:
        Tuple containing two items:
            - A DataFrame containing the synthesized data.
            - The list of keys for the synthesized data.
    """
    _, child_synthesized = sample_from_diffusion(
        df=data,
        df_info=training_results["df_info"],
        diffusion=training_results["diffusion"],
        dataset=training_results["dataset"],
        label_encoders=training_results["label_encoders"],
        sample_size=int(sample_scale * len(data)),
        model_params=ModelParameters(**training_results["model_params"]),
        transformations=Transformations(**training_results["T_dict"]),
        sample_batch_size=sample_batch_size,
    )
    child_keys = list(range(len(child_synthesized)))
    synthesized_final_data = np.concatenate(
        [np.array(child_keys).reshape(-1, 1), child_synthesized.to_numpy()],
        axis=1,
    )
    synthesized_final_df = pd.DataFrame(
        synthesized_final_data,
        columns=[f"{table_name}_id"]
        + training_results["df_info"]["num_cols"]
        + training_results["df_info"]["cat_cols"]
        + [training_results["df_info"]["y_col"]],
    )

    synthesized_final_df = synthesized_final_df[[f"{table_name}_id"] + data.columns.tolist()]

    return synthesized_final_df, child_keys


def _synthesize_multi_table(
    parent_name: str,
    child_name: str,
    parent_training_results: ModelArtifacts,
    child_training_results: ModelArtifacts,
    parent_synthetic_data: ModelArtifacts,
    data: pd.DataFrame,
    group_length_prob_dict: GroupLengthProbDict,
    tables: Tables,
    sample_batch_size: int,
    classifier_scale: float,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Synthesizes data for multi-table using the trained diffusion model and classifier model.

    Args:
        parent_name: Name of the parent table.
        child_name: Name of the child table.
        parent_training_results: Dictionary containing the training results for the parent table.
        child_training_results: Dictionary containing the training results for the child table,
            including the trained diffusion model and the classifier model.
        parent_synthetic_data: Dictionary containing the synthetic data for the parent table.
        data: DataFrame containing the real data to be used for synthesizing.
        group_length_prob_dict: Dictionary containing the group length probabilities for the child and parent tables.
        tables: Tables containing the dataframes and clustering information.
        sample_batch_size: Batch size for sampling.
        classifier_scale: Scale factor for the classifier.

    Returns:
        Tuple containing two items:
            - A DataFrame containing the synthesized data.
            - The list of keys for the synthesized data.
    """
    parent_synthetic_df = parent_synthetic_data["df"]
    parent_keys = parent_synthetic_data["keys"]

    parent_label_index = parent_training_results["column_orders"].index(child_training_results["df_info"]["y_col"])

    parent_synthetic_df_without_id = get_df_without_id(parent_synthetic_df)
    group_labels = parent_synthetic_df_without_id.values[:, parent_label_index].astype(float).astype(int).tolist()

    _, child_synthesized, child_sampled_group_sizes = conditional_sample_from_diffusion(
        df=data,
        df_info=child_training_results["df_info"],
        dataset=child_training_results["dataset"],
        label_encoders=child_training_results["label_encoders"],
        classifier=child_training_results["classifier"],
        diffusion=child_training_results["diffusion"],
        group_labels=group_labels,
        group_length_prob_dict=group_length_prob_dict,
        sample_batch_size=sample_batch_size,
        is_target_conditioned=IsTargetConditioned.NONE,
        classifier_scale=classifier_scale,
    )

    child_foreign_keys = np.repeat(parent_keys, child_sampled_group_sizes, axis=0).reshape((-1, 1))
    child_foreign_keys_arr = np.array(child_foreign_keys).reshape(-1, 1)
    child_primary_keys_arr = np.arange(len(child_synthesized)).reshape(-1, 1)

    child_synthesized_final_arr = np.concatenate(
        [
            child_primary_keys_arr,
            child_synthesized.to_numpy(),
            child_foreign_keys_arr,
        ],
        axis=1,
    )

    child_final_columns = (
        [f"{child_name}_id"]
        + child_training_results["df_info"]["num_cols"]
        + child_training_results["df_info"]["cat_cols"]
        + [child_training_results["df_info"]["y_col"]]
        + [f"{parent_name}_id"]
    )

    child_final_df = pd.DataFrame(child_synthesized_final_arr, columns=child_final_columns)
    original_columns = []
    for col in tables[child_name]["df"].columns:
        if col in child_final_df.columns:
            original_columns.append(col)
    child_final_df = child_final_df[original_columns]

    return child_final_df, child_primary_keys_arr.flatten().tolist()


def _clean_and_save_synthetic_data(
    synthetic_data: dict[str, pd.DataFrame],
    tables: Tables,
    configs: Configs,
) -> dict[str, pd.DataFrame]:
    """
    Cleans the synthetic data by removing the id columns and saving the data to the workspace directory.

    Args:
        synthetic_data: Dictionary containing the synthetic data for each table.
        tables: Dictionary with information about the tables, including the original column names for each table.
        configs: Configuration settings for the workspace directory.

    Returns:
        Dictionary containing the cleaned synthetic data for each table.
    """
    cleaned_synthetic_data: dict[str, pd.DataFrame] = {}
    for table_key, table_val in synthetic_data.items():
        column_names = [column_name for column_name in tables[table_key]["original_cols"] if "_id" not in column_name]
        cleaned_synthetic_data[table_key] = pd.DataFrame(table_val[column_names])

    for cleaned_key, cleaned_val in cleaned_synthetic_data.items():
        table_dir = (
            Path(configs["general"]["workspace_dir"])
            / configs["general"]["exp_name"]
            / cleaned_key
            / f"{configs['general']['sample_prefix']}_final"
        )
        table_dir.mkdir(parents=True, exist_ok=True)
        if f"{cleaned_key}_id" in cleaned_val.columns:
            cleaned_val.to_csv(table_dir / f"{cleaned_key}_synthetic_with_id.csv", index=False)

            val_no_id = cleaned_val.drop(columns=[f"{cleaned_key}_id"])
            val_no_id.to_csv(table_dir / f"{cleaned_key}_synthetic.csv", index=False)
        else:
            cleaned_val.to_csv(table_dir / f"{cleaned_key}_synthetic.csv", index=False)

    return cleaned_synthetic_data
