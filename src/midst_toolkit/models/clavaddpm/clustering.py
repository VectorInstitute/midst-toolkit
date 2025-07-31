import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, QuantileTransformer

from midst_toolkit.models.clavaddpm.params import Configs, RelationOrder, Tables


def clava_clustering(
    tables: Tables,
    relation_order: RelationOrder,
    save_dir: Path,
    configs: Configs,
) -> tuple[Tables, dict[tuple[str, str], dict[int, float]]]:
    """
    Clustering function for the mutli-table function of theClavaDDPM model.

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
        save_dir: Directory to save the clustering checkpoint.
        configs: Dictionary of configurations. The following config keys are required:
            {
                num_clusters = int | dict,
                parent_scale = float,
                clustering_method = str["kmeans" | "both" | "variational" | "gmm"],
            }

    Returns:
        Tuple of the tables and the dictionary of group lengths and probabilities.
    """
    relation_order_reversed = relation_order[::-1]
    all_group_lengths_prob_dicts = {}

    # Clustering
    if os.path.exists(save_dir / "cluster_ckpt.pkl"):
        print("Clustering checkpoint found, loading...")
        cluster_ckpt = pickle.load(open(save_dir / "cluster_ckpt.pkl", "rb"))
        # ruff: noqa: SIM115
        tables = cluster_ckpt["tables"]
        all_group_lengths_prob_dicts = cluster_ckpt["all_group_lengths_prob_dicts"]
    else:
        for parent, child in relation_order_reversed:
            if parent is not None:
                print(f"Clustering {parent} -> {child}")
                if isinstance(configs["num_clusters"], dict):
                    num_clusters = configs["num_clusters"][child]
                else:
                    num_clusters = configs["num_clusters"]
                (
                    parent_df_with_cluster,
                    child_df_with_cluster,
                    group_lengths_prob_dicts,
                ) = _pair_clustering_keep_id(
                    tables[child]["df"],
                    tables[child]["domain"],
                    tables[parent]["df"],
                    tables[parent]["domain"],
                    f"{child}_id",
                    f"{parent}_id",
                    num_clusters,
                    configs["parent_scale"],
                    1,  # not used for now
                    parent,
                    child,
                    clustering_method=configs["clustering_method"],
                )
                tables[parent]["df"] = parent_df_with_cluster
                tables[child]["df"] = child_df_with_cluster
                all_group_lengths_prob_dicts[(parent, child)] = group_lengths_prob_dicts

        cluster_ckpt = {
            "tables": tables,
            "all_group_lengths_prob_dicts": all_group_lengths_prob_dicts,
        }
        pickle.dump(cluster_ckpt, open(save_dir / "cluster_ckpt.pkl", "wb"))
        # ruff: noqa: SIM115

    for parent, child in relation_order:
        if parent is None:
            tables[child]["df"]["placeholder"] = list(range(len(tables[child]["df"])))

    return tables, all_group_lengths_prob_dicts


def _pair_clustering_keep_id(
    # ruff: noqa: PLR0912, PLR0915
    child_df: pd.DataFrame,
    child_domain_dict: dict[str, Any],
    parent_df: pd.DataFrame,
    parent_domain_dict: dict[str, Any],
    child_primary_key: str,
    parent_primary_key: str,
    num_clusters: int,
    parent_scale: float,
    key_scale: float,
    parent_name: str,
    child_name: str,
    clustering_method: Literal["kmeans", "both", "variational", "gmm"] = "kmeans",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict[int, float]]]:
    """
    Pairs clustering information to the parent and child dataframes.

    Used by the mutli-table function of the ClavaDDPM model.

    Args:
        child_df: DataFrame of the child table, as provided by the load_multi_table function.
        child_domain_dict: Dictionary of the child table domain, as provided by the load_multi_table function.
        parent_df: DataFrame of the parent table, as provided by the load_multi_table function.
        parent_domain_dict: Dictionary of the parent table domain, as provided by the load_multi_table function.
        child_primary_key: Name of the child primary key.
        parent_primary_key: Name of the parent primary key.
        num_clusters: Number of clusters.
        parent_scale: Scale of the parent table, provided by the config.
        key_scale: Scale of the key.
        parent_name: Name of the parent table.
        child_name: Name of the child table.
        clustering_method: Method of clustering. Has to be one of ["kmeans", "both", "variational", "gmm"].
            Default is "kmeans".

    Returns:
        Tuple with 3 elements:
            - parent_df_with_cluster: DataFrame of the parent table with the cluster column.
            - child_df_with_cluster: DataFrame of the child table with the cluster column.
            - group_lengths_prob_dicts: Dictionary of group lengths and probabilities.
    """
    original_child_cols = list(child_df.columns)
    original_parent_cols = list(parent_df.columns)

    relation_cluster_name = f"{parent_name}_{child_name}_cluster"

    child_data = child_df.to_numpy()
    parent_data = parent_df.to_numpy()

    child_num_cols = []
    child_cat_cols = []

    parent_num_cols = []
    parent_cat_cols = []

    for col_index, col in enumerate(original_child_cols):
        if col in child_domain_dict:
            if child_domain_dict[col]["type"] == "discrete":
                child_cat_cols.append((col_index, col))
            else:
                child_num_cols.append((col_index, col))

    for col_index, col in enumerate(original_parent_cols):
        if col in parent_domain_dict:
            if parent_domain_dict[col]["type"] == "discrete":
                parent_cat_cols.append((col_index, col))
            else:
                parent_num_cols.append((col_index, col))

    parent_primary_key_index = original_parent_cols.index(parent_primary_key)
    foreing_key_index = original_child_cols.index(parent_primary_key)

    # sort child data by foreign key
    sorted_child_data = child_data[np.argsort(child_data[:, foreing_key_index])]
    child_group_data_dict = _get_group_data_dict(sorted_child_data, [foreing_key_index])

    # sort parent data by primary key
    sorted_parent_data = parent_data[np.argsort(parent_data[:, parent_primary_key_index])]

    group_lengths = []
    unique_group_ids = sorted_parent_data[:, parent_primary_key_index]
    for group_id in unique_group_ids:
        group_id = tuple([group_id])
        # ruff: noqa: C409
        if group_id not in child_group_data_dict:
            group_lengths.append(0)
        else:
            group_lengths.append(len(child_group_data_dict[group_id]))

    group_lengths_np = np.array(group_lengths, dtype=int)

    sorted_parent_data_repeated = np.repeat(sorted_parent_data, group_lengths_np, axis=0)
    assert (sorted_parent_data_repeated[:, parent_primary_key_index] == sorted_child_data[:, foreing_key_index]).all()

    sorted_child_num_data = sorted_child_data[:, [col_index for col_index, col in child_num_cols]]
    sorted_child_cat_data = sorted_child_data[:, [col_index for col_index, col in child_cat_cols]]
    sorted_parent_num_data = sorted_parent_data_repeated[:, [col_index for col_index, col in parent_num_cols]]
    sorted_parent_cat_data = sorted_parent_data_repeated[:, [col_index for col_index, col in parent_cat_cols]]

    joint_num_matrix = np.concatenate([sorted_child_num_data, sorted_parent_num_data], axis=1)
    joint_cat_matrix = np.concatenate([sorted_child_cat_data, sorted_parent_cat_data], axis=1)

    if joint_cat_matrix.shape[1] > 0:
        joint_cat_matrix_p_index = sorted_child_cat_data.shape[1]
        joint_num_matrix_p_index = sorted_child_num_data.shape[1]

        cat_converted = []
        label_encoders = []
        for i in range(joint_cat_matrix.shape[1]):
            # A threshold of 1000 unique values is used to prevent the one-hot encoding of large categorical columns
            if len(np.unique(joint_cat_matrix[:, i])) > 1000:
                continue
            label_encoder = LabelEncoder()
            cat_converted.append(label_encoder.fit_transform(joint_cat_matrix[:, i]).astype(float))
            label_encoders.append(label_encoder)

        cat_converted_transposed = np.vstack(cat_converted).T

        # Initialize an empty array to store the encoded values
        cat_one_hot = np.empty((cat_converted_transposed.shape[0], 0))

        # Loop through each column in the data and encode it
        for col in range(cat_converted_transposed.shape[1]):
            encoder = OneHotEncoder(sparse_output=False)
            column = cat_converted_transposed[:, col].reshape(-1, 1)
            encoded_column = encoder.fit_transform(column)
            cat_one_hot = np.concatenate((cat_one_hot, encoded_column), axis=1)

        cat_one_hot[:, joint_cat_matrix_p_index:] = parent_scale * cat_one_hot[:, joint_cat_matrix_p_index:]

    # Perform quantile normalization using QuantileTransformer
    num_quantile = _quantile_normalize_sklearn(joint_num_matrix)
    num_min_max = _min_max_normalize_sklearn(joint_num_matrix)

    # key_quantile =
    #   quantile_normalize_sklearn(sorted_parent_data_repeated[:, parent_primary_key_index].reshape(-1, 1))
    key_min_max = _min_max_normalize_sklearn(sorted_parent_data_repeated[:, parent_primary_key_index].reshape(-1, 1))

    # key_scaled = key_scaler * key_quantile
    key_scaled = key_scale * key_min_max

    num_quantile[:, joint_num_matrix_p_index:] = parent_scale * num_quantile[:, joint_num_matrix_p_index:]
    num_min_max[:, joint_num_matrix_p_index:] = parent_scale * num_min_max[:, joint_num_matrix_p_index:]

    if joint_cat_matrix.shape[1] > 0:
        cluster_data = np.concatenate((num_min_max, cat_one_hot, key_scaled), axis=1)
    else:
        cluster_data = np.concatenate((num_min_max, key_scaled), axis=1)

    child_group_data = _get_group_data(sorted_child_data, [foreing_key_index])
    child_group_lengths = np.array([len(group) for group in child_group_data], dtype=int)
    num_clusters = min(num_clusters, len(cluster_data))

    # print('clustering')
    if clustering_method == "kmeans":
        kmeans = KMeans(n_clusters=num_clusters, n_init="auto", init="k-means++")
        kmeans.fit(cluster_data)
        cluster_labels = kmeans.labels_
    elif clustering_method == "both":
        gmm = GaussianMixture(
            n_components=num_clusters,
            verbose=1,
            covariance_type="diag",
            init_params="k-means++",
            tol=0.0001,
        )
        gmm.fit(cluster_data)
        cluster_labels = gmm.predict(cluster_data)
    elif clustering_method == "variational":
        gmm = BayesianGaussianMixture(
            n_components=num_clusters,
            verbose=1,
            covariance_type="diag",
            init_params="k-means++",
            tol=0.0001,
        )
        gmm.fit(cluster_data)
        cluster_labels = gmm.predict_proba(cluster_data)
    elif clustering_method == "gmm":
        gmm = GaussianMixture(
            n_components=num_clusters,
            verbose=1,
            covariance_type="diag",
        )
        gmm.fit(cluster_data)
        cluster_labels = gmm.predict(cluster_data)

    if clustering_method == "variational":
        group_cluster_labels, agree_rates = _aggregate_and_sample(cluster_labels, child_group_lengths)
    else:
        # voting to determine the cluster label for each parent
        group_cluster_labels = []
        curr_index = 0
        agree_rates = []
        for group_length in child_group_lengths:
            # First, determine the most common label in the current group
            most_common_label_count = np.max(np.bincount(cluster_labels[curr_index : curr_index + group_length]))
            group_cluster_label = np.argmax(np.bincount(cluster_labels[curr_index : curr_index + group_length]))
            group_cluster_labels.append(int(group_cluster_label))

            # Compute agree rate using the most common label count
            agree_rate = most_common_label_count / group_length
            agree_rates.append(agree_rate)

            # Then, update the curr_index for the next iteration
            curr_index += group_length

    # Compute the average agree rate across all groups
    average_agree_rate = np.mean(agree_rates)
    print("Average agree rate: ", average_agree_rate)

    group_assignment = np.repeat(group_cluster_labels, child_group_lengths, axis=0).reshape((-1, 1))

    # obtain the child data with clustering
    sorted_child_data_with_cluster = np.concatenate([sorted_child_data, group_assignment], axis=1)

    group_labels_list = group_cluster_labels
    group_lengths_list = child_group_lengths.tolist()

    group_lengths_dict: dict[int, dict[int, int]] = {}
    for i in range(len(group_labels_list)):
        group_label = group_labels_list[i]
        if group_label not in group_lengths_dict:
            group_lengths_dict[group_label] = defaultdict(int)
        group_lengths_dict[group_label][group_lengths_list[i]] += 1

    group_lengths_prob_dicts: dict[int, dict[int, float]] = {}
    for group_label, freq_dict in group_lengths_dict.items():
        group_lengths_prob_dicts[group_label] = _freq_to_prob(freq_dict)

    # recover the preprocessed data back to dataframe
    child_df_with_cluster = pd.DataFrame(
        sorted_child_data_with_cluster,
        columns=original_child_cols + [relation_cluster_name],
    )

    # recover child df order
    child_df_with_cluster = pd.merge(
        child_df[[child_primary_key]],
        child_df_with_cluster,
        on=child_primary_key,
        how="left",
    )

    parent_id_to_cluster: dict[Any, Any] = {}
    for i in range(len(sorted_child_data)):
        parent_id = sorted_child_data[i, foreing_key_index]
        if parent_id in parent_id_to_cluster:
            assert parent_id_to_cluster[parent_id] == sorted_child_data_with_cluster[i, -1]
            continue
        parent_id_to_cluster[parent_id] = sorted_child_data_with_cluster[i, -1]

    max_cluster_label = max(parent_id_to_cluster.values())

    parent_data_clusters = []
    for i in range(len(parent_data)):
        if parent_data[i, parent_primary_key_index] in parent_id_to_cluster:
            parent_data_clusters.append(parent_id_to_cluster[parent_data[i, parent_primary_key_index]])
        else:
            parent_data_clusters.append(max_cluster_label + 1)

    parent_data_clusters_np = np.array(parent_data_clusters).reshape(-1, 1)
    parent_data_with_cluster = np.concatenate([parent_data, parent_data_clusters_np], axis=1)
    parent_df_with_cluster = pd.DataFrame(
        parent_data_with_cluster, columns=original_parent_cols + [relation_cluster_name]
    )

    new_col_entry = {
        "type": "discrete",
        "size": len(set(parent_data_clusters_np.flatten())),
    }

    print("Number of cluster centers: ", len(set(parent_data_clusters_np.flatten())))

    parent_domain_dict[relation_cluster_name] = new_col_entry.copy()
    child_domain_dict[relation_cluster_name] = new_col_entry.copy()

    return parent_df_with_cluster, child_df_with_cluster, group_lengths_prob_dicts


def _get_group_data_dict(
    np_data: np.ndarray,
    group_id_attrs: list[int] | None = None,
) -> dict[tuple[Any, ...], list[np.ndarray]]:
    """
    Get the group data dictionary.

    Args:
        np_data: Numpy array of the data.
        group_id_attrs: List of attributes to group by.

    Returns:
        Dictionary of group data.
    """
    if group_id_attrs is None:
        group_id_attrs = [0]

    group_data_dict: dict[tuple[Any, ...], list[np.ndarray]] = {}
    data_len = len(np_data)
    for i in range(data_len):
        row_id = tuple(np_data[i, group_id_attrs])
        if row_id not in group_data_dict:
            group_data_dict[row_id] = []
        group_data_dict[row_id].append(np_data[i])

    return group_data_dict


def _get_group_data(
    np_data: np.ndarray,
    group_id_attrs: list[int] | None = None,
) -> np.ndarray:
    """
    Get the group data.

    Args:
        np_data: Numpy array of the data.
        group_id_attrs: List of attributes to group by.

    Returns:
        Numpy array of the group data.
    """
    if group_id_attrs is None:
        group_id_attrs = [0]

    group_data_list = []
    data_len = len(np_data)
    i = 0
    while i < data_len:
        group = []
        row_id = np_data[i, group_id_attrs]

        while (np_data[i, group_id_attrs] == row_id).all():
            group.append(np_data[i])
            i += 1
            if i >= data_len:
                break
        group_data_list.append(np.array(group))

    return np.array(group_data_list, dtype=object)


# TODO: Refactor the functions below to be a single one with a "method" parameter.


def _quantile_normalize_sklearn(matrix: np.ndarray) -> np.ndarray:
    """
    Quantile normalize the input matrix using Sklearn's QuantileTransformer.

    Args:
        matrix: Numpy array of the matrix data.

    Returns:
        Numpy array of the normalized data.
    """
    transformer = QuantileTransformer(
        output_distribution="normal",
        random_state=42,  # TODO: do we really need to hardcode the random state?
    )  # Change output_distribution as needed

    normalized_data = np.empty((matrix.shape[0], 0))

    # Apply QuantileTransformer to each column and concatenate the results
    for col in range(matrix.shape[1]):
        column = matrix[:, col].reshape(-1, 1)
        transformed_column = transformer.fit_transform(column)
        normalized_data = np.concatenate((normalized_data, transformed_column), axis=1)

    return normalized_data


def _min_max_normalize_sklearn(matrix: np.ndarray) -> np.ndarray:
    """
    Min-max normalize the input matrix using Sklearn's MinMaxScaler.

    Args:
        matrix: Numpy array of the matrix data.

    Returns:
        Numpy array of the normalized data.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))

    normalized_data = np.empty((matrix.shape[0], 0))

    # Apply MinMaxScaler to each column and concatenate the results
    for col in range(matrix.shape[1]):
        column = matrix[:, col].reshape(-1, 1)
        transformed_column = scaler.fit_transform(column)
        normalized_data = np.concatenate((normalized_data, transformed_column), axis=1)

    return normalized_data


def _aggregate_and_sample(
    cluster_probabilities: np.ndarray,
    child_group_lengths: np.ndarray,
) -> tuple[list[int], list[float]]:
    """
    Aggregate the cluster probabilities and sample the labels.

    Used by the variational clustering method.

    Args:
        cluster_probabilities: Numpy array of the cluster probabilities.
        child_group_lengths: Numpy array of the child group lengths.

    Returns:
        Tuple of the group cluster labels and the agree rates.
    """
    group_cluster_labels = []
    curr_index = 0
    agree_rates = []

    for group_length in child_group_lengths:
        # Aggregate the probability distributions by taking the mean
        group_probability_distribution = np.mean(cluster_probabilities[curr_index : curr_index + group_length], axis=0)

        # Sample the label from the aggregated distribution
        group_cluster_label = np.random.choice(
            range(len(group_probability_distribution)), p=group_probability_distribution
        )
        group_cluster_labels.append(group_cluster_label)

        # Compute the max probability as the agree rate
        max_probability = np.max(group_probability_distribution)
        agree_rates.append(max_probability)

        # Update the curr_index for the next iteration
        curr_index += group_length

    return group_cluster_labels, agree_rates


def _freq_to_prob(freq_dict: dict[int, int]) -> dict[int, float]:
    """
    Convert a frequency dictionary to a probability dictionary.

    Args:
        freq_dict: Dictionary of frequencies.

    Returns:
        Dictionary of probabilities.
    """
    prob_dict: dict[Any, float] = {}
    for key, freq in freq_dict.items():
        prob_dict[key] = freq / sum(list(freq_dict.values()))
    return prob_dict
