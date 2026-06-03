import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from torch import Tensor

from midst_toolkit.common.dataset import Dataset, TargetInfo, Transformations
from midst_toolkit.common.dataset_transformations import transform_dataset
from midst_toolkit.common.enumerations import ArrayDict, DataSplit, TaskType


class TabularDataset(Dataset):
    def __init__(self, numerical_features: Tensor, categorical_features: Tensor):
        """Initialize the TabularDataset.

        Args:
            numerical_features: The numerical features.
            categorical_features: The categorical features.
        """
        self.numerical_features_tensor = numerical_features
        self.categorical_features_tensor = categorical_features

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Get the item at the index.

        Args:
            index: The index of the item.

        Returns:
            The item at the index.
        """
        this_num = self.numerical_features_tensor[index]
        this_cat = self.categorical_features_tensor[index]

        return (this_num, this_cat)

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            The length of the dataset.
        """
        return self.numerical_features_tensor.shape[0]


def preprocess(
    dataset_path: Path,
    ref_dataset_path: Path,
    transforms: dict[str, Any],
    task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
    inverse: bool = False,
    concat: bool = True,
) -> Dataset | tuple[ArrayDict, ArrayDict, list[int], int] | tuple[ArrayDict, ArrayDict, list[int], int, Any, Any]:
    """Preprocess the dataset.

    Args:
        dataset_path: The path to the dataset.
        ref_dataset_path: The path to the reference dataset.
        transforms: The transformations to apply to the data.
        task_type: The task type.
        inverse: If True, will also return the inverse of the numerical and
            categorical transformations.
        concat: Whether to concatenate the target to the data.

    Returns:
        The preprocessed dataset,
        OR a tuple containing:
        - The numerical features.
        - The categorical features.
        - The categories.
        - The number of numerical features.
        OR a tuple containing:
        - The numerical features.
        - The categorical features.
        - The categories.
        - The number of numerical features.
        - The numerical inverse transform.
        - The categorical inverse transform.
    """
    transformations = Transformations.from_dict(transforms)
    ref_dataset = make_dataset(
        data_path=ref_dataset_path,
        transformations=transformations,
        task_type=task_type,
        concat=concat,
    )
    assert ref_dataset.numerical_transform is not None, "transform_dataset must be run on ref_dataset"
    assert ref_dataset.categorical_transform is not None, "transform_dataset must be run on ref_dataset"

    dataset = make_dataset(
        data_path=dataset_path,
        transformations=transformations,
        task_type=task_type,
        concat=concat,
    )
    assert dataset.numerical_transform is not None, "transform_dataset must be run on dataset"
    assert dataset.categorical_transform is not None, "transform_dataset must be run on dataset"

    if transformations.categorical_encoding is None:
        assert dataset.numerical_features is not None, "dataset must have numerical features"
        assert dataset.categorical_features is not None, "dataset must have categorical features"
        numerical_features = dataset.numerical_features
        categorical_features = dataset.categorical_features

        numerical_features_train = numerical_features[DataSplit.TRAIN.value]
        numerical_features_test = numerical_features[DataSplit.TEST.value]
        categorical_features_train = categorical_features[DataSplit.TRAIN.value]
        categorical_features_test = categorical_features[DataSplit.TEST.value]

        assert ref_dataset.categorical_features is not None, "ref_dataset must have categorical features"
        ref_categorical_features_train = ref_dataset.categorical_features[DataSplit.TRAIN.value]
        categories = get_categories(ref_categorical_features_train)

        d_numerical = numerical_features_train.shape[1]

        numerical_features = {
            DataSplit.TRAIN.value: numerical_features_train,
            DataSplit.TEST.value: numerical_features_test,
        }
        categorical_features = {
            DataSplit.TRAIN.value: categorical_features_train,
            DataSplit.TEST.value: categorical_features_test,
        }

        if inverse:
            num_inverse = dataset.numerical_transform.inverse_transform
            cat_inverse = ref_dataset.categorical_transform.inverse_transform
            return numerical_features, categorical_features, categories, d_numerical, num_inverse, cat_inverse

        return numerical_features, categorical_features, categories, d_numerical
    return dataset


def make_dataset(
    data_path: Path,
    transformations: Transformations,
    task_type: TaskType,
    concat: bool = True,
) -> Dataset:
    """Make a dataset from the data path.

    Args:
        data_path: The path to the data.
        transformations: The transformations to apply to the data.
        task_type: The task type.
        concat: Whether to concatenate the target to the data.

    Returns:
        The dataset.
    """
    categorical_features: ArrayDict | None = {} if (data_path / "x_cat_train.npy").exists() else None
    numerical_features: ArrayDict | None = {} if (data_path / "x_num_train.npy").exists() else None
    assert (data_path / "y_train.npy").exists(), "y_train.npy does not exist"
    target: ArrayDict = {}

    # classification
    if task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
        for split in [DataSplit.TRAIN, DataSplit.TEST]:
            numerical_features_t, categorical_features_t, target_t = read_pure_data(data_path, split)
            if numerical_features is not None and numerical_features_t is not None:
                numerical_features[split.value] = numerical_features_t
            if categorical_features is not None and categorical_features_t is not None:
                if concat:
                    categorical_features_t = concat_y_to_x(categorical_features_t, target_t)
                categorical_features[split.value] = categorical_features_t
            target[split.value] = target_t
    # regression
    else:
        for split in [DataSplit.TRAIN, DataSplit.TEST]:
            numerical_features_t, categorical_features_t, target_t = read_pure_data(data_path, split)

            if numerical_features is not None and numerical_features_t is not None:
                if concat:
                    numerical_features_t = concat_y_to_x(numerical_features_t, target_t)
                numerical_features[split.value] = numerical_features_t
            if categorical_features is not None and categorical_features_t is not None:
                categorical_features[split.value] = categorical_features_t
            target[split.value] = target_t

    info = json.loads(Path(os.path.join(data_path, "info.json")).read_text())

    dataset = Dataset(
        numerical_features,
        categorical_features,
        target,
        target_info=TargetInfo(),
        task_type=task_type,
        n_classes=info.get("n_classes"),
    )

    return transform_dataset(dataset, transformations, None)


def get_categories(categorical_features_train: np.ndarray | None) -> list[int]:
    """Get the length of the unique categories from the categorical features.

    Args:
        categorical_features_train: The categorical features for the train split.

    Returns:
        The length of the unique categories for each feature.
    """
    if categorical_features_train is None:
        return []
    return [len(set(categorical_features_train[:, i])) for i in range(categorical_features_train.shape[1])]


def read_pure_data(path: Path, split: DataSplit) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Read the pure data from the path.

    Args:
        path: The path to the data.
        split: The split of the data.

    Returns:
        A tuple containing the numerical features, categorical features and target.
    """
    target = np.load(path / f"y_{split.value}.npy", allow_pickle=True)

    numerical_features = None
    if (path / f"x_num_{split.value}.npy").exists():
        numerical_features = np.load(path / f"x_num_{split.value}.npy", allow_pickle=True)

    categorical_features = None
    if (path / f"x_cat_{split.value}.npy").exists():
        categorical_features = np.load(path / f"x_cat_{split.value}.npy", allow_pickle=True)

    return numerical_features, categorical_features, target


def concat_y_to_x(x: np.ndarray | None, y: np.ndarray) -> np.ndarray:
    """Concatenate the target to the input data.

    Args:
        x: The input data. If None, return the target as a column vector.
        y: The target data.

    Returns:
        The concatenated data.
    """
    if x is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), x], axis=1)
