import json
import os
from pathlib import Path

import numpy as np

from midst_toolkit.common.dataset import Dataset, TargetInfo, Transformations
from midst_toolkit.common.dataset_transformations import transform_dataset
from midst_toolkit.common.enumerations import TaskType


class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat):
        self.X_num = X_num
        self.X_cat = X_cat

    def __getitem__(self, index):
        this_num = self.X_num[index]
        this_cat = self.X_cat[index]

        sample = (this_num, this_cat)

        return sample

    def __len__(self):
        return self.X_num.shape[0]


def preprocess(
    dataset_path,
    ref_dataset_path,
    transforms,
    task_type=TaskType.BINARY_CLASSIFICATION,
    inverse=False,
    concat=True,
):
    transformations = Transformations.from_dict(transforms)
    ref_dataset = make_dataset(
        data_path=ref_dataset_path,
        transformations=transformations,
        task_type=task_type,
        change_val=False,
        concat=concat,
    )
    assert ref_dataset.numerical_transform is not None, "transform_dataset must be run on ref_dataset"
    assert ref_dataset.categorical_transform is not None, "transform_dataset must be run on ref_dataset"

    dataset = make_dataset(
        data_path=dataset_path,
        transformations=transformations,
        task_type=task_type,
        change_val=False,
        concat=concat,
    )
    assert dataset.numerical_transform is not None, "transform_dataset must be run on dataset"
    assert dataset.categorical_transform is not None, "transform_dataset must be run on dataset"

    if transformations.categorical_encoding is None:
        X_num = dataset.numerical_features
        X_cat = dataset.categorical_features

        X_train_num, X_test_num = X_num["train"], X_num["test"]
        X_train_cat, X_test_cat = X_cat["train"], X_cat["test"]

        ref_X_train_cat = ref_dataset.categorical_features["train"]
        categories = get_categories(ref_X_train_cat)
        d_numerical = X_train_num.shape[1]

        X_num = (X_train_num, X_test_num)
        X_cat = (X_train_cat, X_test_cat)

        if inverse:
            num_inverse = dataset.numerical_transform.inverse_transform
            # cat_inverse = None
            cat_inverse = ref_dataset.categorical_transform.inverse_transform
            return X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse
        return X_num, X_cat, categories, d_numerical
    return dataset


def make_dataset(
    data_path: str,
    transformations: Transformations,
    task_type: TaskType,
    change_val: bool,
    concat=True,
) -> Dataset:
    # classification
    if task_type == TaskType.BINARY_CLASSIFICATION or task_type == TaskType.MULTICLASS_CLASSIFICATION:
        X_cat = {} if os.path.exists(os.path.join(data_path, "X_cat_train.npy")) else None
        X_num = {} if os.path.exists(os.path.join(data_path, "X_num_train.npy")) else None
        y = {} if os.path.exists(os.path.join(data_path, "y_train.npy")) else None

        for split in ["train", "test"]:
            X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                if concat:
                    X_cat_t = concat_y_to_X(X_cat_t, y_t)
                X_cat[split] = X_cat_t
            if y is not None:
                y[split] = y_t
    else:
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, "X_cat_train.npy")) else None
        X_num = {} if os.path.exists(os.path.join(data_path, "X_num_train.npy")) else None
        y = {} if os.path.exists(os.path.join(data_path, "y_train.npy")) else None

        for split in ["train", "test"]:
            X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)

            if X_num is not None:
                if concat:
                    X_num_t = concat_y_to_X(X_num_t, y_t)
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            if y is not None:
                y[split] = y_t

    info = json.loads(Path(os.path.join(data_path, "info.json")).read_text())

    D = Dataset(
        X_num,
        X_cat,
        y,
        target_info=TargetInfo(),
        task_type=task_type,
        n_classes=info.get("n_classes"),
    )

    if change_val:
        D = change_val(D)

    # def categorical_to_idx(feature):
    #     unique_categories = np.unique(feature)
    #     idx_mapping = {category: index for index, category in enumerate(unique_categories)}
    #     idx_feature = np.array([idx_mapping[category] for category in feature])
    #     return idx_feature

    # for split in ['train', 'val', 'test']:
    # D.y[split] = categorical_to_idx(D.y[split].squeeze(1))

    return transform_dataset(D, transformations, None)


def get_categories(X_train_cat):
    return None if X_train_cat is None else [len(set(X_train_cat[:, i])) for i in range(X_train_cat.shape[1])]


def read_pure_data(path, split="train"):
    y = np.load(os.path.join(path, f"y_{split}.npy"), allow_pickle=True)
    X_num = None
    X_cat = None
    if os.path.exists(os.path.join(path, f"X_num_{split}.npy")):
        X_num = np.load(os.path.join(path, f"X_num_{split}.npy"), allow_pickle=True)
    if os.path.exists(os.path.join(path, f"X_cat_{split}.npy")):
        X_cat = np.load(os.path.join(path, f"X_cat_{split}.npy"), allow_pickle=True)

    return X_num, X_cat, y


def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)
