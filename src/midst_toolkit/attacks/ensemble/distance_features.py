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
        df_input: The dataframe to generate features for (e.g., meta classifier train or test set).
        #TODO: I think it's also worth stating what input is composed of. Is it a mix of things that we can clearly state?
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
    categorical_features = [column in categorical_column_names for column in df_input.columns]

    gower_matrix = gower.gower_matrix(data_x=df_input, data_y=df_synthetic, cat_features=categorical_features)

    # Sort distances for each target record to find nearest neighbors
    sorted_by_distance = np.sort(gower_matrix, axis=1)

    # Create a dictionary to hold new features
    features = {}

    # Min distance and Nearest Neighbor Distance Ratio (NNDR)
    features["min_gower_distance"] = sorted_by_distance[:, 0]

    # NNDR: ratio of the distance to the closest neighbor over the distance to the second closest.
    features["nndr"] = np.divide(
        sorted_by_distance[:, 0],
        sorted_by_distance[:, 1],
        out=np.zeros_like(sorted_by_distance[:, 0]),  # initialize output array with zeros
        where=sorted_by_distance[:, 1] != 0,  # only divide where second min distance is not zero
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
    Computes the DOMIAS (Density-ratio-based Out-of-distribution Model-Inconsistency Assessment Score).

    The score estimates the likelihood that an input sample is an 'overfit' instance
    from the synthetic data, which is rare in the real data distribution. It does so by:
    1. Estimating Densities using Kernel Density Estimation (KDE).
        KDE creates a smooth, non-parametric estimate of the PDF (Probability Density Function)
        for both the real (reference) and synthetic data distributions.
    2. Evaluating Input Points under Both Densities.
        Calculate the estimated probability density for each input sample (x) under:
        - p_ref(x): Real data distribution
        - p_syn(x): Synthetic data distribution
    3. Calculating the Density Ratio for each input sample:
        density_ratio(x) = p_syn(x) / p_ref(x)
        A high ratio implies the point is dense in the synthetic set but sparse in the real set
        (i.e., local overfitting occurred).

    Reference: https://arxiv.org/abs/2302.12580

    Args:
        df_input: The dataframe to calculate DOMIAS scores for (e.g., meta classifier train or test set).
        #TODO: be specific here in the composition of df_input. Is this the challenge dataset or something else.
        df_synthetic: Synthetic data.
        #TODO: Elaborate.
        df_reference: Reference (real) population data.
        #TODO: Ask Fatemeh

    Returns:
        Normalized DOMIAS scores for each test sample, indexed like df_input.
    """
    # Transpose dataframes (.T) to the required (n_features, n_samples) format for scipy's gaussian_kde.
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

    # Scale to [0, 1] (Reshape to 2D for MinMaxScaler, and then flatten back to 1D with ravel())
    pred_proba_domias = MinMaxScaler().fit_transform(density_ratio.reshape(-1, 1)).ravel()

    return pd.DataFrame(pred_proba_domias, columns=["domias"], index=df_input.index)
