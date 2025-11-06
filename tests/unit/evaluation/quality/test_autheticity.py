import numpy as np
import pandas as pd
import pytest

from midst_toolkit.common.random import unset_all_random_seeds
from midst_toolkit.evaluation.quality.alpha_precision import AlphaPrecision
from midst_toolkit.evaluation.utils import one_hot_encode_categoricals_and_merge_with_numerical


def test_autheticity_only_categorical_direct_one_hot():
    categorical_real_data = pd.DataFrame(
        {
            "color": ["red", "blue"],
            "shape": ["circle", "square"],
        }
    )

    categorical_synthetic_data = pd.DataFrame(
        {
            "color": ["red", "blue"],
            "shape": ["square", "circle"],
        }
    )

    categorical_real_encoded = pd.get_dummies(categorical_real_data, columns=["color", "shape"]).astype(int)
    categorical_synthetic_encoded = pd.get_dummies(categorical_synthetic_data, columns=["color", "shape"]).astype(int)
    categorical_synthetic_encoded = categorical_synthetic_encoded.reindex(
        columns=categorical_real_encoded.columns, fill_value=0
    )

    numerical_real_numpy = np.empty((len(categorical_real_data), 0))
    numerical_synthetic_numpy = np.empty((len(categorical_synthetic_data), 0))

    real_dataframe, synthetic_dataframe = one_hot_encode_categoricals_and_merge_with_numerical(
        categorical_real_encoded,
        categorical_synthetic_encoded,
        numerical_real_numpy,
        numerical_synthetic_numpy,
    )

    alpha_precision_metric = AlphaPrecision(naive_only=False)
    quality_results = alpha_precision_metric.compute(real_dataframe, synthetic_dataframe)

    # Check naive authenticity as the _OC metric depends on a 1-layer NN training
    # which may give different results on different architectures
    expected_authenticity = 0.0
    assert pytest.approx(expected_authenticity, abs=1e-8) == quality_results["authenticity_naive"]

    unset_all_random_seeds()


def test_authenticity_only_numerical():
    numerical_real_data = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
        }
    )

    numerical_synthetic_data = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [1.0, 0.0],
        }
    )

    categorical_real_encoded = pd.DataFrame()
    categorical_synthetic_encoded = pd.DataFrame()

    numerical_real_numpy = numerical_real_data.to_numpy()
    numerical_synthetic_numpy = numerical_synthetic_data.to_numpy()

    real_dataframe, synthetic_dataframe = one_hot_encode_categoricals_and_merge_with_numerical(
        categorical_real_encoded,
        categorical_synthetic_encoded,
        numerical_real_numpy,
        numerical_synthetic_numpy,
    )

    alpha_precision_metric = AlphaPrecision(naive_only=False)
    quality_results = alpha_precision_metric.compute(real_dataframe, synthetic_dataframe)

    # Check naive authenticity as the _OC metric depends on a 1-layer NN training
    # which may give different results on different architectures
    expected_authenticity = 0.0
    assert pytest.approx(expected_authenticity, abs=1e-8) == quality_results["authenticity_naive"]

    unset_all_random_seeds()


def test_authenticity_numerical_and_categorical():
    numerical_real_data = pd.DataFrame(
        {
            "num_feature": [0.0, 1.0],
        }
    )

    numerical_synthetic_data = pd.DataFrame(
        {
            "num_feature": [0.0, 1.0],
        }
    )

    categorical_real_data = pd.DataFrame(
        {
            "color": ["red", "blue"],
            "shape": ["circle", "square"],
        }
    )

    categorical_synthetic_data = pd.DataFrame(
        {
            "color": ["red", "blue"],
            "shape": ["square", "circle"],
        }
    )

    categorical_real_encoded = pd.get_dummies(categorical_real_data, columns=["color", "shape"]).astype(int)
    categorical_synthetic_encoded = pd.get_dummies(categorical_synthetic_data, columns=["color", "shape"]).astype(int)
    categorical_synthetic_encoded = categorical_synthetic_encoded.reindex(
        columns=categorical_real_encoded.columns, fill_value=0
    )

    numerical_real_numpy = numerical_real_data.to_numpy()
    numerical_synthetic_numpy = numerical_synthetic_data.to_numpy()

    real_dataframe, synthetic_dataframe = one_hot_encode_categoricals_and_merge_with_numerical(
        categorical_real_encoded,
        categorical_synthetic_encoded,
        numerical_real_numpy,
        numerical_synthetic_numpy,
    )

    alpha_precision_metric = AlphaPrecision(naive_only=False)
    quality_results = alpha_precision_metric.compute(real_dataframe, synthetic_dataframe)

    # Check naive authenticity as the _OC metric depends on a 1-layer NN training
    # which may give different results on different architectures
    expected_authenticity = 0.0
    assert pytest.approx(expected_authenticity, abs=1e-8) == quality_results["authenticity_naive"]

    unset_all_random_seeds()


def test_authenticity_mismatched_sizes_numerical_real_larger():
    numerical_real_data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    numerical_synthetic_data = pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 0.0]})

    categorical_real_encoded = pd.DataFrame()
    categorical_synthetic_encoded = pd.DataFrame()

    numerical_real_numpy = numerical_real_data.to_numpy()
    numerical_synthetic_numpy = numerical_synthetic_data.to_numpy()

    real_dataframe, synthetic_dataframe = one_hot_encode_categoricals_and_merge_with_numerical(
        categorical_real_encoded,
        categorical_synthetic_encoded,
        numerical_real_numpy,
        numerical_synthetic_numpy,
    )

    alpha_precision_metric = AlphaPrecision(naive_only=False)
    quality_results = alpha_precision_metric.compute(real_dataframe, synthetic_dataframe)

    # Check naive authenticity as the _OC metric depends on a 1-layer NN training
    # which may give different results on different architectures
    expected_authenticity = 0.0
    assert pytest.approx(expected_authenticity, abs=1e-8) == quality_results["authenticity_naive"]

    unset_all_random_seeds()


def test_authenticity_mismatched_sizes_numerical_synthetic_larger():
    numerical_real_data = pd.DataFrame(
        {
            "x": [0.0, 2.0],
            "y": [0.0, 2.0],
        }
    )

    numerical_synthetic_data = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 2.0, 3.0],
        }
    )

    categorical_real_encoded = pd.DataFrame()
    categorical_synthetic_encoded = pd.DataFrame()

    numerical_real_numpy = numerical_real_data.to_numpy()
    numerical_synthetic_numpy = numerical_synthetic_data.to_numpy()

    real_dataframe, synthetic_dataframe = one_hot_encode_categoricals_and_merge_with_numerical(
        categorical_real_encoded,
        categorical_synthetic_encoded,
        numerical_real_numpy,
        numerical_synthetic_numpy,
    )

    alpha_precision_metric = AlphaPrecision(naive_only=False)
    quality_results = alpha_precision_metric.compute(real_dataframe, synthetic_dataframe)

    # Check naive authenticity as the _OC metric depends on a 1-layer NN training
    # which may give different results on different architectures
    expected_authenticity = 0.0
    assert pytest.approx(expected_authenticity, abs=1e-8) == quality_results["authenticity_naive"]

    unset_all_random_seeds()
