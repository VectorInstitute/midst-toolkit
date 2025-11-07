from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from midst_toolkit.common.enumerations import TaskType
from midst_toolkit.common.random import set_all_random_seeds, unset_all_random_seeds
from midst_toolkit.models.clavaddpm.dataset import (
    Dataset,
    Transformations,
    get_cached_dataset,
    process_nans_in_numerical_features,
    setup_cache_path,
)
from midst_toolkit.models.clavaddpm.dataset_utils import dump_pickle
from midst_toolkit.models.clavaddpm.enumerations import (
    CategoricalEncoding,
    Normalization,
    NumericalNaNPolicy,
)


def test_load_dataset(tmp_path: Path) -> None:
    train_array = np.random.randn(3, 3)
    val_array = np.random.randn(3, 3)
    test_array = np.random.randn(3, 3)

    np.save(tmp_path / "test_dataset_train.npy", train_array)
    np.save(tmp_path / "test_dataset_val.npy", val_array)
    np.save(tmp_path / "test_dataset_test.npy", test_array)

    # Need to also save label arrays, as thats how the load works...
    np.save(tmp_path / "y_train.npy", np.random.randn(3, 1))
    np.save(tmp_path / "y_val.npy", np.random.randn(3, 1))
    np.save(tmp_path / "y_test.npy", np.random.randn(3, 1))

    datasets = Dataset._load_datasets(tmp_path, "test_dataset")

    assert np.allclose(datasets["train"], train_array, atol=1e-8)
    assert np.allclose(datasets["val"], val_array, atol=1e-8)
    assert np.allclose(datasets["test"], test_array, atol=1e-8)


def _compare_data_splits(test_splits: dict[str, np.ndarray], reference_splits: dict[str, np.ndarray]) -> bool:
    tests = []
    for key, data in test_splits.items():
        tests.append(np.all(data == reference_splits[key]))
    return np.all(tests)


def _get_test_dataset() -> Dataset:
    set_all_random_seeds(42)

    numerical_data_splits = {
        "train": np.random.randint(low=0, high=2, size=(3, 3)).astype(float),
        "val": np.random.randint(low=0, high=2, size=(3, 3)).astype(float),
        "test": np.random.randint(low=0, high=2, size=(3, 3)).astype(float),
    }
    categorical_data_splits = {
        "train": np.random.randint(low=0, high=2, size=(3, 3)).astype(str),
        "val": np.random.randint(low=0, high=2, size=(3, 3)).astype(str),
        "test": np.random.randint(low=0, high=2, size=(3, 3)).astype(str),
    }
    label_splits = {
        "train": np.random.randint(low=0, high=2, size=(3, 3)).astype(int),
        "val": np.random.randint(low=0, high=2, size=(3, 3)).astype(int),
        "test": np.random.randint(low=0, high=2, size=(3, 3)).astype(int),
    }

    dataset = Dataset(
        numerical_features=numerical_data_splits,
        categorical_features=categorical_data_splits,
        target=label_splits,
        target_info={},
        task_type=TaskType.BINARY_CLASSIFICATION,
        n_classes=2,
    )
    unset_all_random_seeds()
    return dataset


def test_process_nans_in_numerical_features_drop() -> None:
    dataset = _get_test_dataset()
    numerical_data_splits = deepcopy(dataset.numerical_features)
    categorical_data_splits = deepcopy(dataset.categorical_features)
    label_splits = deepcopy(dataset.target)

    dataset = process_nans_in_numerical_features(dataset=dataset, policy=NumericalNaNPolicy.DROP_ROWS)
    assert _compare_data_splits(dataset.numerical_features, numerical_data_splits)
    assert _compare_data_splits(dataset.categorical_features, categorical_data_splits)
    assert _compare_data_splits(dataset.target, label_splits)

    # Now add some NaNs to the train and validation splits
    dataset.numerical_features["train"][0, 1] = np.NaN
    dataset.numerical_features["val"][1, 1] = np.NaN
    dataset = process_nans_in_numerical_features(dataset=dataset, policy=NumericalNaNPolicy.DROP_ROWS)
    # Make sure first row of train in all dataset components is dropped
    assert len(dataset.numerical_features["train"]) == 2
    assert len(dataset.categorical_features["train"]) == 2
    assert len(dataset.target["train"]) == 2
    assert np.all(dataset.numerical_features["train"] == numerical_data_splits["train"][1:, :])
    assert np.all(dataset.categorical_features["train"] == categorical_data_splits["train"][1:, :])
    assert np.all(dataset.target["train"] == label_splits["train"][1:, :])
    # Make sure second row of val in all dataset components is dropped
    assert len(dataset.numerical_features["val"]) == 2
    assert len(dataset.categorical_features["val"]) == 2
    assert len(dataset.target["val"]) == 2
    assert np.all(
        dataset.numerical_features["val"]
        == np.vstack((numerical_data_splits["val"][0, :], numerical_data_splits["val"][2, :]))
    )
    assert np.all(
        dataset.categorical_features["val"]
        == np.vstack((categorical_data_splits["val"][0, :], categorical_data_splits["val"][2, :]))
    )
    assert np.all(dataset.target["val"] == np.vstack((label_splits["val"][0, :], label_splits["val"][2, :])))
    assert np.all(dataset.target["test"] == label_splits["test"])

    # Now add NaN to test and make sure we throw.
    dataset.numerical_features["test"][1, 1] = np.NaN

    with pytest.raises(AssertionError):
        dataset = process_nans_in_numerical_features(dataset=dataset, policy=NumericalNaNPolicy.DROP_ROWS)


def test_process_nans_in_numerical_features_mean() -> None:
    dataset = _get_test_dataset()
    numerical_data_splits = deepcopy(dataset.numerical_features)
    categorical_data_splits = deepcopy(dataset.categorical_features)
    label_splits = deepcopy(dataset.target)

    dataset = process_nans_in_numerical_features(dataset=dataset, policy=NumericalNaNPolicy.MEAN)
    assert _compare_data_splits(dataset.numerical_features, numerical_data_splits)
    assert _compare_data_splits(dataset.categorical_features, categorical_data_splits)
    assert _compare_data_splits(dataset.target, label_splits)

    # Now add some NaNs to the train and validation splits
    dataset.numerical_features["train"][0, 1] = np.NaN
    dataset.numerical_features["val"][1, 1] = np.NaN
    # Adding a NaN to a column that doesn't have a NaN in train
    dataset.numerical_features["val"][1, 2] = np.NaN
    dataset = process_nans_in_numerical_features(dataset=dataset, policy=NumericalNaNPolicy.MEAN)
    # Nothing should change in the label and cat rows now
    assert _compare_data_splits(dataset.categorical_features, categorical_data_splits)
    assert _compare_data_splits(dataset.target, label_splits)
    assert dataset.numerical_features["train"][0, 1] == 0
    assert dataset.numerical_features["val"][1, 1] == 0
    assert dataset.numerical_features["val"][1, 2] == 1.0 / 3.0

    # Make sure an error is raised if an entire column is NaN in Train
    with pytest.raises(ValueError):
        dataset.numerical_features["train"][:, 1] = np.NaN
        dataset = process_nans_in_numerical_features(dataset=dataset, policy=NumericalNaNPolicy.MEAN)

    unset_all_random_seeds()


def test_setup_cache_path(tmp_path: Path) -> None:
    transformations_1 = Transformations(seed=2, normalization=Normalization.QUANTILE)
    transformations_2 = Transformations(seed=2, normalization=Normalization.MINMAX)
    transformations_3 = Transformations(seed=2, normalization=Normalization.QUANTILE)
    transformations_4 = Transformations(seed=2, categorical_encoding=CategoricalEncoding.ONE_HOT)

    path_1 = setup_cache_path(transformations_1, tmp_path)
    path_2 = setup_cache_path(transformations_2, tmp_path)
    path_3 = setup_cache_path(transformations_3, tmp_path)
    path_4 = setup_cache_path(transformations_4, tmp_path)
    assert path_1 == path_3
    assert path_1 != path_2
    assert path_1 != path_4

    no_path = setup_cache_path(transformations_1, None)
    assert no_path is None


def test_get_cached_dataset(tmp_path: Path) -> None:
    transformations_1 = Transformations(seed=2, normalization=Normalization.QUANTILE)
    dataset = _get_test_dataset()

    cache_path = setup_cache_path(transformations_1, tmp_path)
    dump_pickle((transformations_1, dataset), cache_path)

    dataset_cache = get_cached_dataset(cache_path, transformations_1)

    assert np.allclose(dataset_cache.numerical_features["train"], dataset.numerical_features["train"], atol=1e-8)
