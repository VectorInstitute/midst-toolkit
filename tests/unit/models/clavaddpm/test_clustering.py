import numpy as np

from midst_toolkit.common.random import set_all_random_seeds, unset_all_random_seeds
from midst_toolkit.models.clavaddpm.clustering import (
    _get_group_data,
    _get_group_data_dict,
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


def test_get_group_data() -> None:
    set_all_random_seeds(42)
    data_array_with_one_foreign_keys = np.hstack(
        (np.random.randn(10, 3), np.random.randint(0, 3, (10, 1)).astype(float), np.random.randn(10, 1))
    )
    data_array_with_two_foreign_keys = np.hstack(
        (np.random.randn(10, 3), np.random.randint(0, 2, (10, 2)).astype(float), np.random.randn(10, 1))
    )
    data_array_with_foreign_key_in_front = np.hstack(
        (np.random.randint(0, 2, (10, 1)).astype(float), np.random.randn(10, 3), np.random.randn(10, 1))
    )

    grouped_data = _get_group_data(data_array_with_one_foreign_keys, [3])
    assert len(grouped_data) == 6
    assert len(grouped_data[0]) == 2
    assert len(grouped_data[1]) == 1
    assert len(grouped_data[2]) == 2
    assert len(grouped_data[5]) == 3
    assert np.allclose(
        grouped_data[0],
        np.array(
            [
                [0.49671415, -0.1382643, 0.64768854, 2.0, 2.77831304],
                [1.52302986, -0.23415337, -0.23413696, 2.0, 1.19363972],
            ]
        ),
        atol=1e-6,
    )
    assert np.allclose(
        grouped_data[1],
        np.array([[1.57921282, 0.76743473, -0.46947439, 0.0, 0.21863832]]),
        atol=1e-6,
    )
    assert np.allclose(
        grouped_data[5],
        np.array(
            [
                [-0.2257763, 0.0675282, -1.42474819, 1.0, -0.53814166],
                [-0.54438272, 0.11092259, -1.15099358, 1.0, -1.3466781],
                [0.37569802, -0.60063869, -0.29169375, 1.0, -0.88059127],
            ]
        ),
        atol=1e-6,
    )

    grouped_data = _get_group_data(data_array_with_one_foreign_keys, None)
    # Entries are unique in the first dimension which is the default for this function.
    assert len(grouped_data) == 10

    grouped_data = _get_group_data(data_array_with_two_foreign_keys, [3, 4])
    assert len(grouped_data) == 6
    assert len(grouped_data[0]) == 3
    assert len(grouped_data[1]) == 2
    assert len(grouped_data[2]) == 1
    assert len(grouped_data[5]) == 1
    assert np.allclose(
        grouped_data[0],
        np.array(
            [
                [-1.1305523, 0.13442888, 0.58212279, 0.0, 0.0, 0.25711687],
                [0.88774846, 0.89433233, 0.7549978, 0.0, 0.0, 0.3145129],
                [-0.20716589, -0.62347739, -1.50815329, 0.0, 0.0, 1.37186213],
            ]
        ),
        atol=1e-6,
    )
    assert np.allclose(
        grouped_data[1],
        np.array(
            [
                [1.09964698, -0.17773212, -0.41038331, 1.0, 0.0, 0.17555329],
                [1.17971634, -0.89820794, 0.83479542, 1.0, 0.0, -0.30928855],
            ]
        ),
        atol=1e-6,
    )
    assert np.allclose(
        grouped_data[2],
        np.array([[0.29656138, -1.03782988, -0.07580375, 0.0, 0.0, 0.6731255]]),
        atol=1e-6,
    )

    grouped_data = _get_group_data(data_array_with_two_foreign_keys, [3])
    assert len(grouped_data) == 5
    assert len(grouped_data[0]) == 3
    assert len(grouped_data[1]) == 2
    assert len(grouped_data[2]) == 1
    assert len(grouped_data[3]) == 2
    assert np.allclose(
        grouped_data[0],
        np.array(
            [
                [-1.1305523, 0.13442888, 0.58212279, 0.0, 0.0, 0.25711687],
                [0.88774846, 0.89433233, 0.7549978, 0.0, 0.0, 0.3145129],
                [-0.20716589, -0.62347739, -1.50815329, 0.0, 0.0, 1.37186213],
            ]
        ),
        atol=1e-6,
    )
    assert np.allclose(
        grouped_data[1],
        np.array(
            [
                [1.09964698, -0.17773212, -0.41038331, 1.0, 0.0, 0.17555329],
                [1.17971634, -0.89820794, 0.83479542, 1.0, 0.0, -0.30928855],
            ]
        ),
        atol=1e-6,
    )
    assert np.allclose(
        grouped_data[3],
        np.array(
            [
                [0.97296353, 0.79559546, 1.49543425, 1.0, 1.0, -0.25663018],
                [0.33818125, 3.37229625, -0.92039081, 1.0, 1.0, -0.36782572],
            ]
        ),
        atol=1e-6,
    )

    grouped_data = _get_group_data(data_array_with_foreign_key_in_front, None)
    # Because the first column is non-unique, we get proper groups.
    assert len(grouped_data) == 5
    assert len(grouped_data[0]) == 3
    assert len(grouped_data[1]) == 2
    assert len(grouped_data[2]) == 2
    assert len(grouped_data[3]) == 1
    assert np.allclose(
        grouped_data[0],
        np.array(
            [
                [0.0, -0.34271452, -0.80227727, -0.16128571, -1.06230371],
                [0.0, 0.40405086, 1.8861859, 0.17457781, 0.47359243],
                [0.0, 0.25755039, -0.07444592, -1.91877122, -0.91942423],
            ]
        ),
        atol=1e-6,
    )
    assert np.allclose(
        grouped_data[1],
        np.array(
            [
                [1.0, -0.02651388, 0.06023021, 2.46324211, 1.54993441],
                [1.0, -0.19236096, 0.30154734, -0.03471177, -0.78325329],
            ]
        ),
        atol=1e-6,
    )
    assert np.allclose(
        grouped_data[3],
        np.array(
            [
                [1.0, -1.40185106, 0.58685709, 2.19045563, -1.23086432],
            ]
        ),
        atol=1e-6,
    )
    unset_all_random_seeds()


def test_get_group_data_dict() -> None:
    set_all_random_seeds(42)
    data_array_with_one_foreign_keys = np.hstack(
        (np.random.randn(10, 3), np.random.randint(0, 3, (10, 1)).astype(float), np.random.randn(10, 1))
    )
    data_array_with_two_foreign_keys = np.hstack(
        (np.random.randn(10, 3), np.random.randint(0, 2, (10, 2)).astype(float), np.random.randn(10, 1))
    )
    data_array_with_foreign_key_in_front = np.hstack(
        (np.random.randint(0, 2, (10, 1)).astype(float), np.random.randn(10, 3), np.random.randn(10, 1))
    )

    grouped_data = _get_group_data_dict(data_array_with_one_foreign_keys, [3])
    assert len(grouped_data) == 3
    assert len(grouped_data[(2.0,)]) == 4
    assert len(grouped_data[(0.0,)]) == 2
    assert np.allclose(
        grouped_data[(0.0,)][0], np.array([1.57921282, 0.76743473, -0.46947439, 0.0, 0.21863832]), atol=1e-6
    )
    assert np.allclose(
        grouped_data[(0.0,)][1], np.array([-0.90802408, -1.4123037, 1.46564877, 0.0, 0.77370042]), atol=1e-6
    )
    assert np.allclose(
        grouped_data[(2.0,)][1], np.array([1.52302986, -0.23415337, -0.23413696, 2.0, 1.19363972]), atol=1e-6
    )

    grouped_data = _get_group_data_dict(data_array_with_one_foreign_keys, None)
    # Entries are unique in the first dimension which is the default for this function.
    assert len(grouped_data) == 10

    grouped_data = _get_group_data_dict(data_array_with_two_foreign_keys, [3, 4])
    assert len(grouped_data) == 4
    assert len(grouped_data[(0.0, 0.0)]) == 5
    assert len(grouped_data[(1.0, 0.0)]) == 2
    assert len(grouped_data[(0.0, 1.0)]) == 1
    assert len(grouped_data[(1.0, 1.0)]) == 2
    assert np.allclose(
        grouped_data[(0.0, 1.0)][0], np.array([-0.39863839, -0.06086409, -1.41875046, 0.0, 1.0, 1.27373362]), atol=1e-6
    )
    assert np.allclose(
        grouped_data[(1.0, 1.0)][0], np.array([0.97296353, 0.79559546, 1.49543425, 1.0, 1.0, -0.25663018]), atol=1e-6
    )
    assert np.allclose(
        grouped_data[(1.0, 1.0)][1], np.array([0.33818125, 3.37229625, -0.92039081, 1.0, 1.0, -0.36782572]), atol=1e-6
    )

    grouped_data = _get_group_data_dict(data_array_with_two_foreign_keys, [3])
    assert len(grouped_data) == 2
    assert len(grouped_data[(0.0,)]) == 6
    assert len(grouped_data[(1.0,)]) == 4
    assert np.allclose(
        grouped_data[(0.0,)][0], np.array([-1.1305523, 0.13442888, 0.58212279, 0.0, 0.0, 0.25711687]), atol=1e-6
    )
    assert np.allclose(
        grouped_data[(0.0,)][2], np.array([-0.20716589, -0.62347739, -1.50815329, 0.0, 0.0, 1.37186213]), atol=1e-6
    )
    assert np.allclose(
        grouped_data[(1.0,)][0], np.array([1.09964698, -0.17773212, -0.41038331, 1.0, 0.0, 0.17555329]), atol=1e-6
    )
    assert np.allclose(
        grouped_data[(1.0,)][3], np.array([0.33818125, 3.37229625, -0.92039081, 1.0, 1.0, -0.36782572]), atol=1e-6
    )

    grouped_data = _get_group_data_dict(data_array_with_foreign_key_in_front, None)
    # Because the first column is non-unique, we get proper groups.
    assert len(grouped_data) == 2
    assert len(grouped_data[(0.0,)]) == 7
    assert len(grouped_data[(1.0,)]) == 3
    unset_all_random_seeds()
