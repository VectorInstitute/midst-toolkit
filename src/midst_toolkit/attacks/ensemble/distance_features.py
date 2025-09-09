# calculate distance features such as Gower distance, DOMIAS, etc.

# FROM GEMINI:

import numpy as np
import pandas as pd
import midst_toolkit.attacks.ensemble.train_utils as gower

from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler


def calculate_gower_features(df_target: pd.DataFrame, df_synth: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """
    Computes Gower distance-based features for a target dataframe against a synthetic one.

    Args:
        df_target: The dataframe to generate features for (e.g., validation or test set).
        df_synth: The synthetic dataframe to compare against.
        cat_cols: A list of categorical column names.

    Returns:
        A dataframe with the new distance-based features, indexed like df_target.
    """
    # Ensure column order matches for distance calculation
    df_synth_aligned = df_synth.reset_index(drop=True)[df_target.columns]

    cat_features_mask = [col in cat_cols for col in df_target.columns]

    # 1. Compute pairwise Gower distance matrix
    pairwise_gower = gower.gower_matrix(data_x=df_target, data_y=df_synth_aligned, cat_features=cat_features_mask)

    # Sort distances for each target record to find nearest neighbors
    dist_sorted = np.sort(pairwise_gower, axis=1)

    # 2. Create a dictionary to hold new features
    features = {}

    # Min distance and Nearest Neighbor Distance Ratio (NNDR)
    features["min_gower_distance"] = dist_sorted[:, 0]
    features["nndr"] = np.divide(
        dist_sorted[:, 0],
        dist_sorted[:, 1],
        out=np.zeros_like(dist_sorted[:, 0]),
        where=dist_sorted[:, 1] != 0,
    )

    # Mean distance to k-nearest neighbors
    for k in [5, 10, 20, 30, 40, 50]:
        features[f"dcr_{k}"] = dist_sorted[:, :k].mean(axis=1)

    # Number of neighbors within a median-based radius (epsilon)
    epsilon = np.median(dist_sorted[:, 0])
    features["num_of_neighbor"] = np.sum(np.where(pairwise_gower <= epsilon, 1, 0), axis=1)

    return pd.DataFrame(features, index=df_target.index)

def domias(df_input: pd.DataFrame, df_synth: pd.DataFrame, df_ref: pd.DataFrame) -> np.ndarray:
    """
    Compute DOMIAS density-ratio-based scores for test data.

    Args:
        df_input: Test data to evaluate (without labels).
        df_synth: Synthetic data.
        df_ref: Reference (real) population data.

    Returns:
        Normalized DOMIAS scores for each test sample, indexed like df_input.
    """
    # Ensure float type and correct orientation for KDE
    ref, synth, input = (df.astype(float).values.T for df in (df_ref, df_synth, df_input))

    # Estimate densities
    density_ref = gaussian_kde(ref)
    density_synth = gaussian_kde(synth)

    # Evaluate test points
    p_ref = density_ref(input)
    p_synth = density_synth(input)

    # Density ratio
    ratio = np.divide(p_synth, p_ref, out=np.zeros_like(p_synth), where=p_ref > 0)

    # Normalize to [0, 1]
    return MinMaxScaler().fit_transform(ratio.reshape(-1, 1)).ravel()