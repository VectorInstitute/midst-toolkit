import numpy as np

from midst_toolkit.common.random import set_all_random_seeds, unset_all_random_seeds
from midst_toolkit.models.clavaddpm.clustering import (
    _min_max_normalize_sklearn,
    _quantile_normalize_sklearn,
    get_normalized_numerical_columns,
    group_data_by_group_id_as_dict,
    group_data_by_id,
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


def test_min_max_normalize_sklearn_empty_matrix() -> None:
    set_all_random_seeds(42)

    data_to_normalize = np.random.randint(0, 3, (5, 0))
    normalized_data = _min_max_normalize_sklearn(data_to_normalize)
    assert data_to_normalize is normalized_data

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


def test_group_data_by_id() -> None:
    set_all_random_seeds(42)
    data_array_with_one_foreign_keys = np.hstack(
        (np.random.randn(10, 3), np.random.randint(0, 3, (10, 1)).astype(float), np.random.randn(10, 1))
    )
    data_array_with_foreign_key_in_front = np.hstack(
        (np.random.randint(0, 2, (10, 1)).astype(float), np.random.randn(10, 3), np.random.randn(10, 1))
    )

    grouped_data = group_data_by_id(data_array_with_one_foreign_keys, 3)
    assert len(grouped_data) == 3
    assert len(grouped_data[0]) == 4
    assert len(grouped_data[1]) == 2
    assert len(grouped_data[2]) == 4
    assert np.allclose(
        grouped_data[0],
        np.array(
            [
                [0.49671415, -0.1382643, 0.64768854, 2.0, 2.77831304],
                [1.52302986, -0.23415337, -0.23413696, 2.0, 1.19363972],
                [0.54256004, -0.46341769, -0.46572975, 2.0, 0.88176104],
                [0.24196227, -1.91328024, -1.72491783, 2.0, -1.00908534],
            ]
        ),
        atol=1e-6,
    )
    assert np.allclose(
        grouped_data[1],
        np.array(
            [
                [1.57921282, 0.76743473, -0.46947439, 0.0, 0.21863832],
                [-0.90802408, -1.4123037, 1.46564877, 0.0, 0.77370042],
            ],
        ),
        atol=1e-6,
    )

    grouped_data = group_data_by_id(data_array_with_foreign_key_in_front, 0, sort_by_column_value=True)
    # Because the first column is non-unique, we get proper groups.
    assert len(grouped_data) == 2
    assert len(grouped_data[0]) == 9
    assert len(grouped_data[1]) == 1
    assert np.allclose(
        grouped_data[1],
        np.array([[1.0, -0.676922, 0.61167629, 1.03099952, 1.47789404]]),
        atol=1e-6,
    )
    assert np.allclose(
        grouped_data[0],
        np.array(
            [
                [0.0, 0.93128012, -0.83921752, -0.30921238, -0.51827022],
                [0.0, 0.33126343, 0.97554513, -0.47917424, -0.8084936],
                [0.0, -0.18565898, -1.10633497, -1.19620662, -0.50175704],
                [0.0, 0.81252582, 1.35624003, -0.07201012, 0.91540212],
                [0.0, 1.0035329, 0.36163603, -0.64511975, 0.32875111],
                [0.0, 0.36139561, 1.53803657, -0.03582604, -0.5297602],
                [0.0, 1.56464366, -2.6197451, 0.8219025, 0.51326743],
                [0.0, 0.08704707, -0.29900735, 0.09176078, 0.09707755],
                [0.0, -1.98756891, -0.21967189, 0.35711257, 0.96864499],
            ]
        ),
        atol=1e-6,
    )
    unset_all_random_seeds()


def test_group_data_by_group_id_as_dict() -> None:
    set_all_random_seeds(42)
    data_array_with_one_foreign_keys = np.hstack(
        (np.random.randn(10, 3), np.random.randint(0, 3, (10, 1)).astype(float), np.random.randn(10, 1))
    )
    data_array_with_foreign_key_in_front = np.hstack(
        (np.random.randint(0, 2, (10, 1)).astype(float), np.random.randn(10, 3), np.random.randn(10, 1))
    )

    grouped_data = group_data_by_group_id_as_dict(data_array_with_one_foreign_keys, 3)
    assert len(grouped_data) == 3
    assert len(grouped_data[2]) == 4
    assert len(grouped_data[0]) == 2
    assert np.allclose(grouped_data[0][0], np.array([1.57921282, 0.76743473, -0.46947439, 0.0, 0.21863832]), atol=1e-6)
    assert np.allclose(grouped_data[0][1], np.array([-0.90802408, -1.4123037, 1.46564877, 0.0, 0.77370042]), atol=1e-6)
    assert np.allclose(
        grouped_data[2][1], np.array([1.52302986, -0.23415337, -0.23413696, 2.0, 1.19363972]), atol=1e-6
    )

    grouped_data = group_data_by_group_id_as_dict(data_array_with_foreign_key_in_front, 0)
    # Because the first column is non-unique, we get proper groups.
    assert len(grouped_data) == 2
    assert len(grouped_data[0]) == 9
    assert len(grouped_data[1]) == 1
    unset_all_random_seeds()
