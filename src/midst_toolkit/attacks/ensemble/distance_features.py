"""Calculate distance features such as Gower distance, DOMIAS, etc."""

import gower
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler


def calculate_gower_features(
    df_input: pd.DataFrame, df_synthetic: pd.DataFrame, categorical_column_names: list[str]
) -> pd.DataFrame:
    """
    Computes Gower distance-based features for a target dataframe against a synthetic one.

    Args:
        df_input: The dataframe to generate features for (e.g., meta classifier train set).
        df_synthetic: The synthetic dataframe to compare against.
        categorical_column_names: A list of categorical column names.

    Returns:
        A dataframe with shape (num_samples, 9) with the new distance-based features, indexed like df_input.
        The 9 features include:
            - min_gower_distance: Minimum Gower distance to any synthetic point.
            - nndr: Nearest Neighbor Distance Ratio (min distance / second min distance).
            - dcr_k: Mean distance to the k-nearest neighbors (for k in {5, 10, 20, 30, 40, 50}).
            - num_of_neighbor: Number of synthetic neighbors within a median-based radius.
    """
    categorical_features = [col in categorical_column_names for col in df_input.columns]

    gower_matrix = gower.gower_matrix(data_x=df_input, data_y=df_synthetic, cat_features=categorical_features)

    # Sort distances for each target record to find nearest neighbors
    sorted_by_distance = np.sort(gower_matrix, axis=1)

    # Create a dictionary to hold new features
    features = {}

    # Min distance and Nearest Neighbor Distance Ratio (NNDR)
    features["min_gower_distance"] = sorted_by_distance[:, 0]
    features["nndr"] = np.divide(
        sorted_by_distance[:, 0],
        sorted_by_distance[:, 1],
        out=np.zeros_like(sorted_by_distance[:, 0]),
        where=sorted_by_distance[:, 1] != 0,
    )

    # Mean distance to k-nearest neighbors
    for k in [5, 10, 20, 30, 40, 50]:
        features[f"dcr_{k}"] = sorted_by_distance[:, :k].mean(axis=1)

    # Number of neighbors within a median-based radius (epsilon)
    epsilon = np.median(sorted_by_distance[:, 0])
    features["num_of_neighbor"] = np.sum(np.where(gower_matrix <= epsilon, 1, 0), axis=1)

    return pd.DataFrame(features, index=df_input.index)


def calculate_domias_score(
    df_input: pd.DataFrame, df_synthetic: pd.DataFrame, df_reference: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute DOMIAS density-ratio-based scores for test data.

    Args:
        df_input: Test data to evaluate (without labels).
        df_synthetic: Synthetic data.
        df_reference: Reference (real) population data.

    Returns:
        Normalized DOMIAS scores for each test sample, indexed like df_input.
    """
    # Ensure float type and correct orientation for KDE
    reference_data_transposed, synthetic_data_transposed, input_data_transposed = (
        df.astype(float).values.T for df in (df_reference, df_synthetic, df_input)
    )

    # Estimate densities
    reference_data_density = gaussian_kde(reference_data_transposed)
    synthetic_data_density = gaussian_kde(synthetic_data_transposed)

    # Evaluate input points under both densities
    reference_data_probability = reference_data_density(input_data_transposed)
    synthetic_data_probability = synthetic_data_density(input_data_transposed)

    # Density ratio. The higher the ratio, the more likely the point is synthetic (not in the real data)
    density_ratio = np.divide(
        synthetic_data_probability,
        reference_data_probability,
        out=np.zeros_like(synthetic_data_probability),
        where=reference_data_probability > 0,
    )

    # Scale to [0, 1]
    pred_proba_domias = MinMaxScaler().fit_transform(density_ratio.reshape(-1, 1)).ravel()

    return pd.DataFrame(pred_proba_domias, columns=["domias"], index=df_input.index)
