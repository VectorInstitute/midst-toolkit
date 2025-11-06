import numpy as np

from midst_toolkit.common.random import set_all_random_seeds, unset_all_random_seeds
from midst_toolkit.models.clavaddpm.clustering import (
    _min_max_normalize_sklearn,
    _quantile_normalize_sklearn,
    get_normalized_numerical_columns,
)
from midst_toolkit.models.clavaddpm.enumerations import DataAndKeyNormalizationType


def test_quantile_normalize_sklearn() -> None:
    set_all_random_seeds(42)
    data_to_normalize = np.random.randint(0, 3, (5, 5))
    normalized_data = _quantile_normalize_sklearn(data_to_normalize)
    assert np.allclose(
        normalized_data,
        np.array(
            [
                [5.19933758, -5.19933758, 5.19933758, 5.19933758, -5.19933758],
                [-5.19933758, 5.19933758, 0.0, 5.19933758, 5.19933758],
                [5.19933758, 5.19933758, -5.19933758, 5.19933758, 0.31863936],
                [-5.19933758, 0.0, 0.0, -5.19933758, 0.31863936],
                [-5.19933758, -5.19933758, 0.0, -5.19933758, -5.19933758],
            ]
        ),
        atol=1e-5,
    )
    unset_all_random_seeds()


def test_min_max_normalize_sklearn() -> None:
    set_all_random_seeds(42)
    data_to_normalize = np.random.randint(0, 3, (5, 5))
    normalized_data = _min_max_normalize_sklearn(data_to_normalize)
    assert np.allclose(
        normalized_data,
        np.array(
            [
                [1.0, -1.0, 1.0, 1.0, -1.0],
                [-1.0, 1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, -1.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, -1.0, 0.0],
                [-1.0, -1.0, 0.0, -1.0, -1.0],
            ]
        ),
        atol=1e-8,
    )
    unset_all_random_seeds()


def test_get_normalized_numerical_columns() -> None:
    set_all_random_seeds(42)
    child_data = np.random.randint(0, 3, (3, 3))
    parent_data = np.random.randint(0, 3, (3, 3))
    scale = 2.0
    normalization_type = DataAndKeyNormalizationType.MINMAX

    normalized_data = get_normalized_numerical_columns(child_data, parent_data, scale, normalization_type)
    assert np.allclose(
        normalized_data,
        np.array(
            [[-1.0, -1.0, 1.0, 2.0, 2.0, 2.0], [-1.0, -1.0, -1.0, -2.0, 2.0, -2.0], [-1.0, 1.0, 1.0, -2.0, -2.0, -2.0]]
        ),
        atol=1e-6,
    )

    normalization_type = DataAndKeyNormalizationType.QUANTILE
    normalized_data = get_normalized_numerical_columns(child_data, parent_data, scale, normalization_type)

    assert np.allclose(
        normalized_data,
        np.array(
            [
                [-5.19933758, -5.19933758, 5.19933758, 2 * 5.19933758, 2 * 5.19933758, 2 * 5.19933758],
                [-5.19933758, -5.19933758, -5.19933758, 2 * -5.19933758, 2 * 5.19933758, 2 * -5.19933758],
                [-5.19933758, 5.19933758, 5.19933758, 2 * -5.19933758, 2 * -5.19933758, 2 * -5.19933758],
            ]
        ),
        atol=1e-5,
    )

    unset_all_random_seeds()
