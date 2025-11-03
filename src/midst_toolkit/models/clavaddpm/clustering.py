"""Clustering functions for the multi-table ClavaDDPM model."""

import os
import pickle
from collections import defaultdict
from logging import INFO, WARNING
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, QuantileTransformer

from midst_toolkit.common.config import ClusteringConfig
from midst_toolkit.common.enumerations import DomainDataType
from midst_toolkit.common.logger import log
from midst_toolkit.models.clavaddpm.enumerations import (
    ClusteringMethod,
    GroupLengthsProbDicts,
    KeyScalingType,
    RelationOrder,
    Tables,
)


def clava_clustering(
    tables: Tables,
    relation_order: RelationOrder,
    save_dir: Path,
    configs: ClusteringConfig,
) -> tuple[dict[str, Any], GroupLengthsProbDicts]:
    """
    Clustering function for the multi-table function of the ClavaDDPM model.

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
        configs: Configuration for the clustering model.

    Returns:
        A tuple with 2 values:
            - The tables dictionary.
            - The dictionary with the group lengths probability for all the parent-child pairs.
    """
    cluster_ckpt = _load_clustering_info_from_checkpoint(save_dir)
    if cluster_ckpt is not None:
        tables = cluster_ckpt["tables"]
        all_group_lengths_prob_dicts = cluster_ckpt["all_group_lengths_prob_dicts"]

    else:
        tables, all_group_lengths_prob_dicts = _run_clustering(tables, relation_order, configs)

        # saving the clustering information in the checkpoint file
        cluster_ckpt = {
            "tables": tables,
            "all_group_lengths_prob_dicts": all_group_lengths_prob_dicts,
        }
        with open(save_dir / "cluster_ckpt.pkl", "wb") as f:
            pickle.dump(cluster_ckpt, f)

    # adding a placeholder for the top level tables (i.e. tables with no parent)
    for parent, child in relation_order:
        if parent is None:
            tables[child]["df"]["placeholder"] = list(range(len(tables[child]["df"])))

    return tables, all_group_lengths_prob_dicts


def _load_clustering_info_from_checkpoint(save_dir: Path) -> dict[str, Any] | None:
    """
    Load the clustering information from the checkpoint if it exists.

    Args:
        save_dir: Directory to save the clustering checkpoint.

    Returns:
        Clustering information as a dictionary if the checkpoint exists, None otherwise.
        The dictionary contains the tables under the "tables" key and the group lengths
        probabilities under the "all_group_lengths_prob_dicts" key.
    """
    if not os.path.exists(save_dir / "cluster_ckpt.pkl"):
        return None

    log(INFO, "Clustering checkpoint found, loading...")

    with open(save_dir / "cluster_ckpt.pkl", "rb") as f:
        return pickle.load(f)


def _run_clustering(
    tables: Tables,
    relation_order: RelationOrder,
    configs: ClusteringConfig,
) -> tuple[Tables, GroupLengthsProbDicts]:
    """
    Run the clustering process.

    Args:
        tables: Dictionary of the tables by name.
        relation_order: List of tuples of parent and child tables. Example:
            [("table1", "table2"), ("table1", "table3")]
        configs: Configuration for the clustering model.

    Returns:
        Tuple with 2 elements:
            - The tables dictionary.
            - The dictionary with the group lengths probability for all the parent-child pairs.
    """
    all_group_lengths_prob_dicts = {}
    relation_order_reversed = relation_order[::-1]
    for parent, child in relation_order_reversed:
        if parent is not None:
            log(INFO, f"Clustering {parent} -> {child}")
            if isinstance(configs.num_clusters, dict):
                num_clusters = configs.num_clusters[child]
            else:
                num_clusters = configs.num_clusters
            (
                parent_df_with_cluster,
                child_df_with_cluster,
                group_lengths_prob_dicts,
            ) = _pair_clustering(
                tables,
                child,
                parent,
                num_clusters,
                configs.parent_scale,
                1,  # not used for now
                clustering_method=configs.clustering_method,
            )
            tables[parent]["df"] = parent_df_with_cluster
            tables[child]["df"] = child_df_with_cluster
            all_group_lengths_prob_dicts[(parent, child)] = group_lengths_prob_dicts

    return tables, all_group_lengths_prob_dicts


def _pair_clustering(
    tables: Tables,
    child_name: str,
    parent_name: str,
    num_clusters: int,
    parent_scale: float,
    key_scale: float,
    clustering_method: ClusteringMethod = ClusteringMethod.KMEANS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict[int, float]]]:
    """
    Pairs clustering information to the parent and child dataframes.

    Used by the mutli-table function of the ClavaDDPM model.

    Args:
        tables: Dictionary of the tables by name.
        parent_name: Name of the parent table.
        child_name: Name of the child table.
        num_clusters: Number of clusters.
        parent_scale: Scaling factor applied to the parent table, provided by the config.
            It will be applied to the features to weight their importance during clustering.
        key_scale: Scaling factor applied to the foreign key values that link
            the child table to the parent table. This will weight how much influence
            the parent-child relationship has in the clustering algorithm.
        clustering_method: Method of clustering. Default is ClusteringMethod.KMEANS.

    Returns:
        Tuple with 3 elements:
            - parent_df_with_cluster: DataFrame of the parent table with the cluster column.
            - child_df_with_cluster: DataFrame of the child table with the cluster column.
            - group_lengths_prob_dicts: Dictionary of group lengths and probabilities.

        NOTE: It will also mutate the "domain" dictionaries under the child and parent tables
        to add the following entry:
            "{parent_name}_{child_name}_cluster": {
                "type": "discrete",
                "size": num_clusters,
            }
    """
    child_df = tables[child_name]["df"]
    parent_df = tables[parent_name]["df"]
    # The domain dictionary holds metadata about the columns of each one of the tables.
    child_domain = tables[child_name]["domain"]
    parent_domain = tables[parent_name]["domain"]
    child_primary_key = f"{child_name}_id"
    parent_primary_key = f"{parent_name}_id"
    all_child_columns = list(child_df.columns)
    all_parent_columns = list(parent_df.columns)

    parent_primary_key_index = all_parent_columns.index(parent_primary_key)
    foreign_key_index = all_child_columns.index(parent_primary_key)

    # sort child data by foreign key
    child_data = child_df.to_numpy()
    sorted_child_data = child_data[np.argsort(child_data[:, foreign_key_index])]

    # sort parent data by primary key
    parent_data = parent_df.to_numpy()
    sorted_parent_data = parent_data[np.argsort(parent_data[:, parent_primary_key_index])]

    cluster_data = _prepare_cluster_data(
        sorted_child_data,
        sorted_parent_data,
        child_domain,
        parent_domain,
        all_child_columns,
        all_parent_columns,
        parent_primary_key,
        parent_scale,
        key_scale,
    )

    cluster_labels = _get_cluster_labels(cluster_data, clustering_method, num_clusters)

    child_group_data = _get_group_data(sorted_child_data, [foreign_key_index])
    child_group_lengths = np.array([len(group) for group in child_group_data], dtype=int)

    if clustering_method == ClusteringMethod.VARIATIONAL:
        group_cluster_labels, agree_rates = _aggregate_and_sample(cluster_labels, child_group_lengths)
    else:
        group_cluster_labels, agree_rates = _get_group_cluster_labels_through_voting(
            cluster_labels, child_group_lengths
        )

    # Compute the average agree rate across all groups
    average_agree_rate = np.mean(agree_rates)
    log(INFO, f"Average agree rate: {average_agree_rate}")

    # obtain the child data with clustering
    group_assignment = np.repeat(group_cluster_labels, child_group_lengths, axis=0).reshape((-1, 1))
    sorted_child_data_with_cluster = np.concatenate([sorted_child_data, group_assignment], axis=1)

    # recover the preprocessed data back to dataframe
    relation_cluster_name = f"{parent_name}_{child_name}_cluster"
    child_df_with_cluster = pd.DataFrame(
        sorted_child_data_with_cluster,
        columns=all_child_columns + [relation_cluster_name],
    )

    # recover child df order
    child_df_with_cluster = pd.merge(
        child_df[[child_primary_key]],
        child_df_with_cluster,
        on=child_primary_key,
        how="left",
    )

    parent_data_clusters = _get_parent_data_clusters(
        sorted_child_data_with_cluster,
        parent_data,
        parent_primary_key_index,
        foreign_key_index,
    )
    parent_data_with_cluster = np.concatenate([parent_data, parent_data_clusters], axis=1)
    parent_df_with_cluster = pd.DataFrame(
        parent_data_with_cluster, columns=all_parent_columns + [relation_cluster_name]
    )

    group_lengths_probabilities = _get_group_lengths_probabilities(
        group_cluster_labels,
        child_group_lengths.tolist(),
    )

    new_col_entry = {
        "type": DomainDataType.DISCRETE.value,
        "size": len(set(parent_data_clusters.flatten())),
    }

    log(INFO, f"Number of cluster centers: {new_col_entry['size']}")

    parent_domain[relation_cluster_name] = new_col_entry.copy()
    child_domain[relation_cluster_name] = new_col_entry.copy()

    return parent_df_with_cluster, child_df_with_cluster, group_lengths_probabilities


def _merge_parent_data_with_child_data(
    child_data: np.ndarray,
    parent_data: np.ndarray,
    parent_primary_key_index: int,
    foreign_key_index: int,
) -> np.ndarray:
    """
    Merge the parent data in relation to the child group data.

    This is done by duplicating the parent data for each element of the child group data
    in a process akin to database table denormalization.

    https://en.wikipedia.org/wiki/Denormalization

    Args:
        child_data: Numpy array of the child data. Should be sorted by the foreign key.
        parent_data: Numpy array of the parent data. Should be sorted by the parent primary key.
        parent_primary_key_index: Index of the parent primary key.
        foreign_key_index: Index of the foreign key to the child data.

    Returns:
        Numpy array of the parent data merged for each group of the child group data.
    """
    child_group_data_dict = _get_group_data_dict(child_data, [foreign_key_index])

    group_lengths = []
    unique_group_ids = parent_data[:, parent_primary_key_index]
    for group_id in unique_group_ids:
        group_id_tuple = (group_id,)
        if group_id_tuple not in child_group_data_dict:
            group_lengths.append(0)
        else:
            group_lengths.append(len(child_group_data_dict[group_id_tuple]))
    group_lengths_np = np.array(group_lengths, dtype=int)
    merged_parent_data = np.repeat(parent_data, group_lengths_np, axis=0)
    assert (merged_parent_data[:, parent_primary_key_index] == child_data[:, foreign_key_index]).all()

    return merged_parent_data


def _get_min_max_and_quantile_for_numerical_columns(
    child_numerical_data: np.ndarray,
    parent_numerical_data: np.ndarray,
    parent_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the min-max and quantile values for the numerical columns in both the
    child and parent data.

    Args:
        child_numerical_data: Numpy array of the child numerical data.
        parent_numerical_data: Numpy array of the parent numerical data.
        parent_scale: Scaling factor applied to the parent data.

    Returns:
        A tuple with two numpy arrays, one with the min-max values and one with the quantile
        values for the numerical columns.
    """
    joint_matrix = np.concatenate([child_numerical_data, parent_numerical_data], axis=1)
    matrix_p_index = child_numerical_data.shape[1]

    # Perform quantile normalization using QuantileTransformer
    numerical_quantile = _quantile_normalize_sklearn(joint_matrix)
    numerical_min_max = _min_max_normalize_sklearn(joint_matrix)

    numerical_quantile[:, matrix_p_index:] = parent_scale * numerical_quantile[:, matrix_p_index:]
    numerical_min_max[:, matrix_p_index:] = parent_scale * numerical_min_max[:, matrix_p_index:]

    return numerical_min_max, numerical_quantile


def _one_hot_encode_categorical_columns(
    child_categorical_data: np.ndarray,
    parent_categorical_data: np.ndarray,
    parent_scale: float,
) -> np.ndarray | None:
    """
    One-hot encode the categorical columns in both the child and parent data.

    Args:
        child_categorical_data: Numpy array of the child categorical data.
        parent_categorical_data: Numpy array of the parent categorical data.
        parent_scale: Scaling factor applied to the parent data.

    Returns:
        Numpy array of the one-hot encoded categorical columns.
    """
    joint_matrix = np.concatenate([child_categorical_data, parent_categorical_data], axis=1)
    if joint_matrix.shape[1] == 0:
        return None

    matrix_p_index = child_categorical_data.shape[1]

    categories_converted = []
    for i in range(joint_matrix.shape[1]):
        # A threshold of 1000 unique values is used to prevent the one-hot encoding of large categorical columns
        if len(np.unique(joint_matrix[:, i])) > 1000:
            log(WARNING, f"Categorical column '{i}' has more than 1000 unique values, skipping...")
            continue

        categories_converted.append(LabelEncoder().fit_transform(joint_matrix[:, i]).astype(float))

    transposed_categories = np.vstack(categories_converted).T

    # Initialize an empty array to store the encoded values
    categorical_one_hot = np.empty((transposed_categories.shape[0], 0))

    # Loop through each column in the data and encode it
    for column in range(transposed_categories.shape[1]):
        encoder = OneHotEncoder(sparse_output=False)
        reshaped_column = transposed_categories[:, column].reshape(-1, 1)
        encoded_column = encoder.fit_transform(reshaped_column)
        categorical_one_hot = np.concatenate((categorical_one_hot, encoded_column), axis=1)

    categorical_one_hot[:, matrix_p_index:] = parent_scale * categorical_one_hot[:, matrix_p_index:]

    return categorical_one_hot


def _prepare_cluster_data(
    child_data: np.ndarray,
    parent_data: np.ndarray,
    child_domain: dict[str, Any],
    parent_domain: dict[str, Any],
    all_child_columns: list[str],
    all_parent_columns: list[str],
    parent_primary_key: str,
    parent_scale: float,
    key_scale: float,
    key_scaling_type: KeyScalingType = KeyScalingType.MINMAX,
) -> np.ndarray:
    """
    Prepare the data for the clustering algorithm, which comprises of merging the parent
    and child data, splitting the data into categorical and numerical columns, and
    normalizing the data.

    Args:
        child_data: Numpy array of the child data.
        parent_data: Numpy array of the parent data.
        child_domain: Dictionary of the domain of the child table. The domain dictionary
            holds metadata about the columns of each one of the tables.
        parent_domain: Dictionary of the domain of the parent table. The domain dictionary
            holds metadata about the columns of each one of the tables.
        all_child_columns: List of all child columns.
        all_parent_columns: List of all parent columns.
        parent_primary_key: Name of the parent primary key.
        parent_scale: Scaling factor applied to the parent table, provided by the config.
            It will be applied to the features to weight their importance during clustering.
        key_scale: Scaling factor applied to the tables' keys. This will weight how much influence
            the parent-child relationship has in the clustering algorithm.
        key_scaling_type: Type of scaling for the tables' keys. Default is KeyScalingType.MINMAX.

    Returns:
        Numpy array of the data prepared for the clustering algorithm.
    """
    # Recalculating the keys' indices here to save us from passing one extra parameter.
    parent_primary_key_index = all_parent_columns.index(parent_primary_key)
    foreign_key_index = all_child_columns.index(parent_primary_key)

    merged_data = _merge_parent_data_with_child_data(
        child_data,
        parent_data,
        parent_primary_key_index,
        foreign_key_index,
    )

    # Splitting the data columns into categorical and numerical based on the domain dictionary.
    # Columns that are not in the domain dictionary are ignored (except for the primary and foreign keys).
    child_numerical_columns, child_categorical_columns = _get_categorical_and_numerical_columns(
        all_child_columns,
        child_domain,
    )
    parent_numerical_columns, parent_categorical_columns = _get_categorical_and_numerical_columns(
        all_parent_columns,
        parent_domain,
    )

    child_numerical_data = child_data[:, child_numerical_columns]
    child_categorical_data = child_data[:, child_categorical_columns]
    parent_numerical_data = merged_data[:, parent_numerical_columns]
    parent_categorical_data = merged_data[:, parent_categorical_columns]

    numerical_min_max, numerical_quantile = _get_min_max_and_quantile_for_numerical_columns(
        child_numerical_data,
        parent_numerical_data,
        parent_scale,
    )

    reshaped_parent_data = merged_data[:, parent_primary_key_index].reshape(-1, 1)
    if key_scaling_type == KeyScalingType.MINMAX:
        key_normalized = _min_max_normalize_sklearn(reshaped_parent_data)
        numerical_normalized = numerical_min_max
    elif key_scaling_type == KeyScalingType.QUANTILE:
        key_normalized = _quantile_normalize_sklearn(reshaped_parent_data)
        numerical_normalized = numerical_quantile
    else:
        raise ValueError(f"Unsupported foreign key scaling type: {key_scaling_type}")

    key_scaled = key_scale * key_normalized

    categorical_one_hot = _one_hot_encode_categorical_columns(
        child_categorical_data,
        parent_categorical_data,
        parent_scale,
    )

    if categorical_one_hot is None:
        return np.concatenate((numerical_normalized, key_scaled), axis=1)

    return np.concatenate((numerical_normalized, categorical_one_hot, key_scaled), axis=1)


def _get_cluster_labels(
    cluster_data: np.ndarray,
    clustering_method: ClusteringMethod,
    num_clusters: int,
) -> np.ndarray:
    """
    Get the cluster labels from the clustering algorithm chosen by the given clustering method.
    The cluster labels are obtained by fitting the clustering algorithm to the data prepared
    for the clustering algorithm.

    Args:
        cluster_data: Numpy array of the data prepared for the clustering algorithm.
        clustering_method: The clustering method to use.
        num_clusters: Number of clusters. If the number of clusters is greater than the
            number of data points, the number of clusters will be set to the number of data points.

    Returns:
        Numpy array of the cluster labels for the data.
    """
    num_clusters = min(num_clusters, len(cluster_data))

    if clustering_method == ClusteringMethod.KMEANS:
        kmeans = KMeans(n_clusters=num_clusters, n_init="auto", init="k-means++")
        kmeans.fit(cluster_data)
        cluster_labels = kmeans.labels_
    elif clustering_method == ClusteringMethod.KMEANS_AND_GMM:
        gmm = GaussianMixture(
            n_components=num_clusters,
            verbose=1,
            covariance_type="diag",
            init_params="k-means++",
            tol=0.0001,
        )
        gmm.fit(cluster_data)
        cluster_labels = gmm.predict(cluster_data)
    elif clustering_method == ClusteringMethod.VARIATIONAL:
        bgmm = BayesianGaussianMixture(
            n_components=num_clusters,
            verbose=1,
            covariance_type="diag",
            init_params="k-means++",
            tol=0.0001,
        )
        bgmm.fit(cluster_data)
        cluster_labels = bgmm.predict_proba(cluster_data)
    elif clustering_method == ClusteringMethod.GMM:
        gmm = GaussianMixture(
            n_components=num_clusters,
            verbose=1,
            covariance_type="diag",
        )
        gmm.fit(cluster_data)
        cluster_labels = gmm.predict(cluster_data)

    return cluster_labels


def _get_group_lengths_probabilities(
    group_cluster_labels: list[int],
    child_group_lengths: list[int],
) -> dict[int, dict[int, float]]:
    """
    Calculate the group lengths probabilities from the frequency in which the child group lengths
    appear for each of the group cluster labels.

    Args:
        group_cluster_labels: List of the group cluster labels.
        child_group_lengths: List of the child group lengths.

    Returns:
        Dictionary of the group lengths probabilities.
        The keys are the group cluster labels and the values are the probabilities of the group lengths.
    """
    group_lengths_dict: dict[int, dict[int, int]] = {}
    for i in range(len(group_cluster_labels)):
        group_label = group_cluster_labels[i]
        if group_label not in group_lengths_dict:
            group_lengths_dict[group_label] = defaultdict(int)
        group_lengths_dict[group_label][child_group_lengths[i]] += 1

    group_lengths_probabilities: dict[int, dict[int, float]] = {}
    for group_label, frequencies_dict in group_lengths_dict.items():
        group_lengths_probabilities[group_label] = _frequency_to_probability(frequencies_dict)

    return group_lengths_probabilities


def _get_parent_data_clusters(
    child_data_with_cluster: np.ndarray,
    parent_data: np.ndarray,
    parent_primary_key_index: int,
    foreign_key_index: int,
) -> np.ndarray:
    """
    Get the parent data clusters from the child data with cluster and the parent data.
    The child data needs to be sorted by the foreign key.

    Args:
        child_data_with_cluster: Numpy array of the child data with cluster information.
            Should be sorted by the foreign key.
        parent_data: Numpy array of the parent data.
        parent_primary_key_index: Index of the parent primary key.
        foreign_key_index: Index of the foreign key to the child data.

    Returns:
        Numpy array of the parent data clusters.
    """
    parent_id_to_cluster: dict[Any, Any] = {}
    for i in range(len(child_data_with_cluster)):
        parent_id = child_data_with_cluster[i, foreign_key_index]
        if parent_id in parent_id_to_cluster:
            assert parent_id_to_cluster[parent_id] == child_data_with_cluster[i, -1]
        else:
            parent_id_to_cluster[parent_id] = child_data_with_cluster[i, -1]

    max_cluster_label = max(parent_id_to_cluster.values())

    parent_data_clusters = []
    for i in range(len(parent_data)):
        if parent_data[i, parent_primary_key_index] in parent_id_to_cluster:
            parent_data_clusters.append(parent_id_to_cluster[parent_data[i, parent_primary_key_index]])
        else:
            parent_data_clusters.append(max_cluster_label + 1)

    return np.array(parent_data_clusters).reshape(-1, 1)


def _get_categorical_and_numerical_columns(
    all_columns: list[str],
    table_domain: dict[str, Any],
) -> tuple[list[int], list[int]]:
    """
    Return the list of numerical and categorical column indices from the table domain dictionary.

    Args:
        all_columns: List of all columns.
        table_domain: Dictionary of the table's domain containing metadata about the data columns.

    Returns:
        Tuple with two lists of indices, one for the numerical columns and one for the categorical columns.
    """
    numerical_columns = []
    categorical_columns = []

    for col_index, column in enumerate(all_columns):
        if column in table_domain:
            if table_domain[column]["type"] == DomainDataType.DISCRETE.value:
                categorical_columns.append(col_index)
            else:
                numerical_columns.append(col_index)

    return numerical_columns, categorical_columns


def _get_group_data_dict(
    np_data: np.ndarray,
    group_id_attrs: list[int] | None = None,
) -> dict[tuple[Any, ...], list[np.ndarray]]:
    """
    Group rows in a numpy array by their values in specified grouping columns into a dictionary.
    Returns a dict where keys are tuples of grouping values and values are lists of corresponding rows.

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
    Group consecutive rows in a numpy array based on specified grouping attributes.
    Returns an array of arrays where each sub-array contains rows with identical
    values in the grouping columns.

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

        # TODO refactor this condition to be more readable/understandable.
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


def _get_group_cluster_labels_through_voting(
    cluster_labels: np.ndarray,
    child_group_lengths: np.ndarray,
) -> tuple[list[int], list[float]]:
    """
    Get the group cluster labels through voting.

    Used by the non-variational clustering methods.

    Args:
        cluster_labels: Numpy array of the cluster labels.
        child_group_lengths: Numpy array of the child group lengths.

    Returns:
        Tuple of the group cluster labels and the agree rates.
    """
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

    return group_cluster_labels, agree_rates


def _frequency_to_probability(frequencies: dict[int, int]) -> dict[int, float]:
    """
    Convert a frequency dictionary to a probability dictionary.

    Args:
        frequencies: Dictionary of frequencies.

    Returns:
        Dictionary of probabilities.
    """
    probabilities: dict[Any, float] = {}
    for key, freq in frequencies.items():
        probabilities[key] = freq / sum(list(frequencies.values()))
    return probabilities
