import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from midst_toolkit.common.dataset import Dataset, TargetInfo, Transformations
from midst_toolkit.common.dataset_transformations import transform_dataset
from midst_toolkit.common.enumerations import ArrayDict, DataSplit, TaskType


class TabularDataset(Dataset):
    def __init__(self, X_num: ArrayDict, X_cat: ArrayDict):
        self.X_num = X_num
        self.X_cat = X_cat

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        this_num = self.X_num[index]
        this_cat = self.X_cat[index]

        sample = (this_num, this_cat)

        return sample

    def __len__(self):
        return self.X_num.shape[0]


def preprocess(
    dataset_path: Path,
    ref_dataset_path: Path,
    transforms: dict[str, Any],
    task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
    inverse: bool = False,
    concat: bool = True,
) -> (
    Dataset
    | tuple[ArrayDict, ArrayDict, list[int] | None, int]
    | tuple[ArrayDict, ArrayDict, list[int] | None, int, Any, Any]
):
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
        X_num = dataset.numerical_features
        X_cat = dataset.categorical_features

        X_train_num, X_test_num = X_num[DataSplit.TRAIN.value], X_num[DataSplit.TEST.value]
        X_train_cat, X_test_cat = X_cat[DataSplit.TRAIN.value], X_cat[DataSplit.TEST.value]

        assert ref_dataset.categorical_features is not None, "ref_dataset must have categorical features"
        ref_X_train_cat = ref_dataset.categorical_features[DataSplit.TRAIN.value]
        categories = get_categories(ref_X_train_cat)

        d_numerical = X_train_num.shape[1]

        X_num = {DataSplit.TRAIN.value: X_train_num, DataSplit.TEST.value: X_test_num}
        X_cat = {DataSplit.TRAIN.value: X_train_cat, DataSplit.TEST.value: X_test_cat}

        if inverse:
            num_inverse = dataset.numerical_transform.inverse_transform
            cat_inverse = ref_dataset.categorical_transform.inverse_transform
            return X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse

        return X_num, X_cat, categories, d_numerical
    return dataset


def make_dataset(
    data_path: Path,
    transformations: Transformations,
    task_type: TaskType,
    concat: bool = True,
) -> Dataset:
    X_cat: ArrayDict | None = {} if (data_path / "X_cat_train.npy").exists() else None
    X_num: ArrayDict | None = {} if (data_path / "X_num_train.npy").exists() else None
    assert (data_path / "y_train.npy").exists(), "y_train.npy does not exist"
    y: ArrayDict = {}

    # classification
    if task_type == TaskType.BINARY_CLASSIFICATION or task_type == TaskType.MULTICLASS_CLASSIFICATION:
        for split in [DataSplit.TRAIN, DataSplit.TEST]:
            X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)
            if X_num is not None and X_num_t is not None:
                X_num[split.value] = X_num_t
            if X_cat is not None and X_cat_t is not None:
                if concat:
                    X_cat_t = concat_y_to_X(X_cat_t, y_t)
                X_cat[split.value] = X_cat_t
            y[split.value] = y_t
    # regression
    else:
        for split in [DataSplit.TRAIN, DataSplit.TEST]:
            X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)

            if X_num is not None and X_num_t is not None:
                if concat:
                    X_num_t = concat_y_to_X(X_num_t, y_t)
                X_num[split.value] = X_num_t
            if X_cat is not None and X_cat_t is not None:
                X_cat[split.value] = X_cat_t
            y[split.value] = y_t

    info = json.loads(Path(os.path.join(data_path, "info.json")).read_text())

    dataset = Dataset(
        X_num,
        X_cat,
        y,
        target_info=TargetInfo(),
        task_type=task_type,
        n_classes=info.get("n_classes"),
    )

    return transform_dataset(dataset, transformations, None)


def get_categories(X_train_cat: np.ndarray | None) -> list[int] | None:
    return None if X_train_cat is None else [len(set(X_train_cat[:, i])) for i in range(X_train_cat.shape[1])]


def read_pure_data(path: Path, split: DataSplit) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    y = np.load(path / f"y_{split.value}.npy", allow_pickle=True)

    X_num = None
    if (path / f"X_num_{split.value}.npy").exists():
        X_num = np.load(path / f"X_num_{split.value}.npy", allow_pickle=True)

    X_cat = None
    if (path / f"X_cat_{split.value}.npy").exists():
        X_cat = np.load(path / f"X_cat_{split.value}.npy", allow_pickle=True)

    return X_num, X_cat, y


def concat_y_to_X(X: np.ndarray | None, y: np.ndarray) -> np.ndarray:
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)
