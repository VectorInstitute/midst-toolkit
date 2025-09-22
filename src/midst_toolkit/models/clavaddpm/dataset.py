"""Defines the dataset functions for the ClavaDDPM model."""

import hashlib
import json
import pickle
from collections import Counter
from copy import deepcopy
from dataclasses import astuple, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Self, cast

import numpy as np
import pandas as pd
import torch
from category_encoders import LeaveOneOutEncoder
from scipy.special import expit, softmax
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileTransformer,
    StandardScaler,
)

from midst_toolkit.models.clavaddpm.typing import ArrayDict, IsYCond


CAT_MISSING_VALUE = "__nan__"
CAT_RARE_VALUE = "__rare__"


Normalization = Literal["standard", "quantile", "minmax"]
NumNanPolicy = Literal["drop-rows", "mean"]
CatNanPolicy = Literal["most_frequent"]
CatEncoding = Literal["one-hot", "counter"]
YPolicy = Literal["default"]


class TaskType(Enum):
    BINCLASS = "binclass"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"

    def __str__(self) -> str:
        """
        Return the string representation of the task type, which is the value of the enum.

        Returns:
            The string representation of the task type.
        """
        return self.value


class PredictionType(Enum):
    LOGITS = "logits"
    PROBS = "probs"


@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Normalization | None = None
    num_nan_policy: NumNanPolicy | None = None
    cat_nan_policy: CatNanPolicy | None = None
    cat_min_frequency: float | None = None
    cat_encoding: CatEncoding | None = None
    y_policy: YPolicy | None = "default"

    @classmethod
    def default(cls) -> Self:
        """Return the default transformations."""
        return cls(
            seed=0,
            normalization="quantile",
            num_nan_policy=None,
            cat_nan_policy=None,
            cat_min_frequency=None,
            cat_encoding=None,
            y_policy="default",
        )


@dataclass(frozen=False)
class Dataset:
    X_num: ArrayDict | None
    X_cat: ArrayDict | None
    y: ArrayDict
    y_info: dict[str, Any]
    task_type: TaskType
    n_classes: int | None
    cat_transform: OneHotEncoder | None = None
    num_transform: StandardScaler | None = None

    @classmethod
    def from_dir(cls, dir_: Path | str) -> Self:
        """
        Load a dataset from a directory.

        Args:
            dir_: The directory to load the dataset from. Can be a Path object or a path string.

        Returns:
            The loaded dataset.
        """
        dir_ = Path(dir_)
        splits = [k for k in ["train", "val", "test"] if dir_.joinpath(f"y_{k}.npy").exists()]

        def load(item: str) -> ArrayDict:
            return {x: cast(np.ndarray, np.load(dir_ / f"{item}_{x}.npy", allow_pickle=True)) for x in splits}

        if Path(dir_ / "info.json").exists():
            info = json.loads(Path(dir_ / "info.json").read_text())

        return cls(
            load("X_num") if dir_.joinpath("X_num_train.npy").exists() else None,
            load("X_cat") if dir_.joinpath("X_cat_train.npy").exists() else None,
            load("y"),
            {},
            TaskType(info["task_type"]),
            info.get("n_classes"),
        )

    @property
    def is_binclass(self) -> bool:
        """
        Check if the dataset is a binary classification dataset.

        Returns:
            True if the dataset is a binary classification dataset, False otherwise.
        """
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        """
        Check if the dataset is a multiclass classification dataset.

        Returns:
            True if the dataset is a multiclass classification dataset, False otherwise.
        """
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        """
        Check if the dataset is a regression dataset.

        Returns:
            True if the dataset is a regression dataset, False otherwise.
        """
        return self.task_type == TaskType.REGRESSION

    @property
    def n_num_features(self) -> int:
        """
        Get the number of numerical features in the dataset.

        Returns:
            The number of numerical features in the dataset.
        """
        return 0 if self.X_num is None else self.X_num["train"].shape[1]

    @property
    def n_cat_features(self) -> int:
        """
        Get the number of categorical features in the dataset.

        Returns:
            The number of categorical features in the dataset.
        """
        return 0 if self.X_cat is None else self.X_cat["train"].shape[1]

    @property
    def n_features(self) -> int:
        """
        Get the total number of features in the dataset.

        Returns:
            The total number of features in the dataset.
        """
        return self.n_num_features + self.n_cat_features

    # TODO: make partition into an Enum
    def size(self, partition: Literal["train", "val", "test"] | None) -> int:
        """
        Get the size of the dataset.

        Args:
            partition: The partition of the dataset to get the size of.
                If None, the size of the entire dataset is returned.

        Returns:
            The size of the dataset.
        """
        return sum(map(len, self.y.values())) if partition is None else len(self.y[partition])

    @property
    def nn_output_dim(self) -> int:
        """
        Get the output dimension of the neural network.

        Returns:
            The output dimension of the neural network.
        """
        if self.is_multiclass:
            assert self.n_classes is not None
            return self.n_classes
        return 1

    def get_category_sizes(self, partition: Literal["train", "val", "test"]) -> list[int]:
        """
        Get the size of the categories in the dataset.

        Args:
            partition: The partition of the dataset to get the size of the categories of.

        Returns:
            The size of the categories in the partition of the dataset.
        """
        return [] if self.X_cat is None else get_category_sizes(self.X_cat[partition])

    # TODO: prediciton_type should be of type PredictionType
    def calculate_metrics(
        self,
        predictions: dict[str, np.ndarray],
        prediction_type: str | PredictionType | None,
    ) -> dict[str, Any]:
        """
        Calculate the metrics of the predictions.

        Args:
            predictions: The predictions to calculate the metrics of.
            prediction_type: The type of the predictions.

        Returns:
            The metrics of the predictions.
        """
        metrics = {
            x: calculate_metrics(self.y[x], predictions[x], self.task_type, prediction_type, self.y_info)
            for x in predictions
        }
        if self.task_type == TaskType.REGRESSION:
            score_key = "rmse"
            score_sign = -1
        else:
            score_key = "accuracy"
            score_sign = 1
        for part_metrics in metrics.values():
            part_metrics["score"] = score_sign * part_metrics[score_key]
        return metrics


# TODO consider moving all the functions below into the Dataset class
def get_category_sizes(X: torch.Tensor | np.ndarray) -> list[int]:
    """
    Get the size of the categories in the data.

    Args:
        X: The data to get the size of the categories of.

    Returns:
        A list with the category sizes in the data.
    """
    XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
    return [len(set(x)) for x in XT]


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str | TaskType,
    prediction_type: str | PredictionType | None,
    y_info: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate the metrics of the predictions.

    Usage: calculate_metrics(y_true, y_pred, 'binclass', 'logits', {})

    Args:
        y_true: The true labels as a numpy array.
        y_pred: The predicted labels as a numpy array.
        task_type: The type of the task.
        prediction_type: The type of the predictions.
        y_info: A dictionary with metadata about the labels.

    Returns:
        The metrics of the predictions.
    """
    task_type = TaskType(task_type)
    if prediction_type is not None:
        prediction_type = PredictionType(prediction_type)

    if task_type == TaskType.REGRESSION:
        assert prediction_type is None
        assert "std" in y_info
        rmse = calculate_rmse(y_true, y_pred, y_info["std"])
        r2 = r2_score(y_true, y_pred)
        result = {"rmse": rmse, "r2": r2}
    else:
        labels, probs = _get_labels_and_probs(y_pred, task_type, prediction_type)
        result = cast(dict[str, Any], classification_report(y_true, labels, output_dict=True))
        if task_type == TaskType.BINCLASS:
            result["roc_auc"] = roc_auc_score(y_true, probs)
    return result


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray, std: float | None) -> float:
    """
    Calculate the root mean squared error (RMSE) of the predictions.

    Args:
        y_true: The true labels as a numpy array.
        y_pred: The predicted labels as a numpy array.
        std: The standard deviation of the labels. If None, the RMSE is calculated
            without the standard deviation.

    Returns:
        The RMSE of the predictions.
    """
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    if std is not None:
        rmse *= std
    return rmse


def _get_labels_and_probs(
    y_pred: np.ndarray, task_type: TaskType, prediction_type: PredictionType | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Get the labels and probabilities from the predictions.

    Args:
        y_pred: The predicted labels as a numpy array.
        task_type: The type of the task.
        prediction_type: The type of the predictions.

    Returns:
        A tuple with the labels and probabilities. The probabilities are None
            if the prediction_type is None.
    """
    assert task_type in (TaskType.BINCLASS, TaskType.MULTICLASS)

    if prediction_type is None:
        return y_pred, None

    if prediction_type == PredictionType.LOGITS:
        probs = expit(y_pred) if task_type == TaskType.BINCLASS else softmax(y_pred, axis=1)
    elif prediction_type == PredictionType.PROBS:
        probs = y_pred
    else:
        raise ValueError(f"Unknown prediction_type: {prediction_type}")

    assert probs is not None
    labels = np.round(probs) if task_type == TaskType.BINCLASS else probs.argmax(axis=1)
    return labels.astype("int64"), probs


def make_dataset_from_df(
    # ruff: noqa: PLR0915, PLR0912
    df: pd.DataFrame,
    transformations: Transformations,
    is_y_cond: IsYCond,
    df_info: pd.DataFrame,
    ratios: list[float] | None = None,
    std: float = 0,
) -> tuple[Dataset, dict[int, LabelEncoder], list[int]]:
    """
    Generate a dataset from a pandas DataFrame.

    The order of the generated dataset: (y, X_num, X_cat).

    Note: For now, n_classes has to be set to 0. This is because our matrix is the concatenation
    of (X_num, X_cat). In this case, if we have is_y_cond == 'concat', we can guarantee that y
    is the first column of the matrix.
    However, if we have n_classes > 0, then y is not the first column of the matrix.

    Args:
        df: The pandas DataFrame to generate the dataset from.
        transformations: The transformations to apply to the dataset.
        is_y_cond: The condition on the y column.
            concat: y is concatenated to X, the model learn a joint distribution of (y, X)
            embedding: y is not concatenated to X. During computations, y is embedded
                and added to the latent vector of X
            none: y column is completely ignored

            How does is_y_cond affect the generation of y?
            is_y_cond:
                concat: the model synthesizes (y, X) directly, so y is just the first column
                embedding: y is first sampled using empirical distribution of y. The model only
                    synthesizes X. When returning the generated data, we return the generated X
                    and the sampled y. (y is sampled from empirical distribution, instead of being
                    generated by the model)
                    Note that in this way, y is still not independent of X, because the model has been
                    adding the embedding of y to the latent vector of X during computations.
                none:
                    y is synthesized using y's empirical distribution. X is generated by the model.
                    In this case, y is completely independent of X.

        df_info: A dictionary with metadata about the DataFrame.
        ratios: The ratios of the dataset to split into train, val, and test. Optional, default is [0.7, 0.2, 0.1].
        std: The standard deviation of the labels. Optional, default is 0.

    Returns:
        A tuple with the dataset, the label encoders, and the column orders.
    """
    if ratios is None:
        ratios = [0.7, 0.2, 0.1]

    train_val_df, test_df = train_test_split(df, test_size=ratios[2], random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=ratios[1] / (ratios[0] + ratios[1]), random_state=42)

    cat_column_orders = []
    num_column_orders = []
    index_to_column = list(df.columns)
    column_to_index = {col: i for i, col in enumerate(index_to_column)}

    if df_info["n_classes"] > 0:
        X_cat: dict[str, np.ndarray] | None = (
            {} if df_info["cat_cols"] is not None or is_y_cond == IsYCond.CONCAT else None
        )
        X_num: dict[str, np.ndarray] | None = {} if df_info["num_cols"] is not None else None
        y = {}

        cat_cols_with_y = []
        if df_info["cat_cols"] is not None:
            cat_cols_with_y += df_info["cat_cols"]
        if is_y_cond == IsYCond.CONCAT:
            cat_cols_with_y = [df_info["y_col"]] + cat_cols_with_y

        if len(cat_cols_with_y) > 0:
            X_cat["train"] = train_df[cat_cols_with_y].to_numpy(dtype=np.str_)  # type: ignore[index]
            X_cat["val"] = val_df[cat_cols_with_y].to_numpy(dtype=np.str_)  # type: ignore[index]
            X_cat["test"] = test_df[cat_cols_with_y].to_numpy(dtype=np.str_)  # type: ignore[index]

        y["train"] = train_df[df_info["y_col"]].values.astype(np.float32)
        y["val"] = val_df[df_info["y_col"]].values.astype(np.float32)
        y["test"] = test_df[df_info["y_col"]].values.astype(np.float32)

        if df_info["num_cols"] is not None:
            X_num["train"] = train_df[df_info["num_cols"]].values.astype(np.float32)  # type: ignore[index]
            X_num["val"] = val_df[df_info["num_cols"]].values.astype(np.float32)  # type: ignore[index]
            X_num["test"] = test_df[df_info["num_cols"]].values.astype(np.float32)  # type: ignore[index]

        cat_column_orders = [column_to_index[col] for col in cat_cols_with_y]
        num_column_orders = [column_to_index[col] for col in df_info["num_cols"]]

    else:
        X_cat = {} if df_info["cat_cols"] is not None else None
        X_num = {} if df_info["num_cols"] is not None or is_y_cond == IsYCond.CONCAT else None
        y = {}

        num_cols_with_y = []
        if df_info["num_cols"] is not None:
            num_cols_with_y += df_info["num_cols"]
        if is_y_cond == IsYCond.CONCAT:
            num_cols_with_y = [df_info["y_col"]] + num_cols_with_y

        if len(num_cols_with_y) > 0:
            X_num["train"] = train_df[num_cols_with_y].values.astype(np.float32)  # type: ignore[index]
            X_num["val"] = val_df[num_cols_with_y].values.astype(np.float32)  # type: ignore[index]
            X_num["test"] = test_df[num_cols_with_y].values.astype(np.float32)  # type: ignore[index]

        y["train"] = train_df[df_info["y_col"]].values.astype(np.float32)
        y["val"] = val_df[df_info["y_col"]].values.astype(np.float32)
        y["test"] = test_df[df_info["y_col"]].values.astype(np.float32)

        if df_info["cat_cols"] is not None:
            X_cat["train"] = train_df[df_info["cat_cols"]].to_numpy(dtype=np.str_)  # type: ignore[index]
            X_cat["val"] = val_df[df_info["cat_cols"]].to_numpy(dtype=np.str_)  # type: ignore[index]
            X_cat["test"] = test_df[df_info["cat_cols"]].to_numpy(dtype=np.str_)  # type: ignore[index]

        cat_column_orders = [column_to_index[col] for col in df_info["cat_cols"]]
        num_column_orders = [column_to_index[col] for col in num_cols_with_y]

    column_orders = num_column_orders + cat_column_orders
    column_orders = [index_to_column[index] for index in column_orders]

    label_encoders = {}
    if X_cat is not None and len(df_info["cat_cols"]) > 0:
        X_cat_all = np.vstack((X_cat["train"], X_cat["val"], X_cat["test"]))
        X_cat_converted = []
        for col_index in range(X_cat_all.shape[1]):
            label_encoder = LabelEncoder()
            X_cat_converted.append(label_encoder.fit_transform(X_cat_all[:, col_index]).astype(float))
            if std > 0:
                # add noise
                X_cat_converted[-1] += np.random.normal(0, std, X_cat_converted[-1].shape)
            label_encoders[col_index] = label_encoder

        X_cat_converted = np.vstack(X_cat_converted).T  # type: ignore[assignment]

        train_num = X_cat["train"].shape[0]
        val_num = X_cat["val"].shape[0]
        # test_num = X_cat["test"].shape[0]

        X_cat["train"] = X_cat_converted[:train_num, :]  # type: ignore[call-overload]
        X_cat["val"] = X_cat_converted[train_num : train_num + val_num, :]  # type: ignore[call-overload]
        X_cat["test"] = X_cat_converted[train_num + val_num :, :]  # type: ignore[call-overload]

        if X_num and len(X_num) > 0:
            X_num["train"] = np.concatenate((X_num["train"], X_cat["train"]), axis=1)
            X_num["val"] = np.concatenate((X_num["val"], X_cat["val"]), axis=1)
            X_num["test"] = np.concatenate((X_num["test"], X_cat["test"]), axis=1)
        else:
            X_num = X_cat
            X_cat = None

    D = Dataset(
        # ruff: noqa: N806
        X_num,
        None,
        y,
        y_info={},
        task_type=TaskType(df_info["task_type"]),
        n_classes=df_info["n_classes"],
    )

    return transform_dataset(D, transformations, None), label_encoders, column_orders


def transform_dataset(
    dataset: Dataset,
    transformations: Transformations,
    cache_dir: Path | None,
) -> Dataset:
    """
    Transform the dataset.

    Args:
        dataset: The dataset to transform.
        transformations: The transformations to apply to the dataset.
        cache_dir: The directory to cache the transformed dataset.
            Optional, default is None. If not None, will check if the transformations exist in the cache directory.
            If they do, will returned the cached transformed dataset. If not, will transform the dataset and cache it.

    Returns:
        The transformed dataset.
    """
    # WARNING: the order of transformations matters. Moreover, the current
    # implementation is not ideal in that sense.
    cache_path = None
    if cache_dir is not None:
        # if cache_dir is not None, will save the cahe file path into the cache_path variable
        # so the transformations can be saved in the cache dir
        transformations_md5 = hashlib.md5(str(transformations).encode("utf-8")).hexdigest()
        transformations_str = "__".join(map(str, astuple(transformations)))
        cache_path = cache_dir / f"cache__{transformations_str}__{transformations_md5}.pickle"
        if cache_path.exists():
            cache_transformations, value = load_pickle(cache_path)
            if transformations == cache_transformations:
                print(f"Using cached features: {cache_dir.name + '/' + cache_path.name}")
                return value
            raise RuntimeError(f"Hash collision for {cache_path}")

    if dataset.X_num is not None:
        dataset = num_process_nans(dataset, transformations.num_nan_policy)

    num_transform = None
    cat_transform = None
    X_num = dataset.X_num

    if X_num is not None and transformations.normalization is not None:
        X_num, num_transform = normalize(  # type: ignore[assignment]
            X_num,
            transformations.normalization,
            transformations.seed,
            return_normalizer=True,
        )

    if dataset.X_cat is None:
        assert transformations.cat_nan_policy is None
        assert transformations.cat_min_frequency is None
        # assert transformations.cat_encoding is None
        X_cat = None
    else:
        X_cat = cat_process_nans(dataset.X_cat, transformations.cat_nan_policy)
        if transformations.cat_min_frequency is not None:
            X_cat = cat_drop_rare(X_cat, transformations.cat_min_frequency)
        X_cat, is_num, cat_transform = cat_encode(
            X_cat,
            transformations.cat_encoding,
            dataset.y["train"],
            transformations.seed,
            return_encoder=True,
        )
        if is_num:
            X_num = X_cat if X_num is None else {x: np.hstack([X_num[x], X_cat[x]]) for x in X_num}
            X_cat = None

    y, y_info = build_target(dataset.y, transformations.y_policy, dataset.task_type)

    dataset = replace(dataset, X_num=X_num, X_cat=X_cat, y=y, y_info=y_info)
    dataset.num_transform = num_transform
    dataset.cat_transform = cat_transform

    if cache_path is not None:
        dump_pickle((transformations, dataset), cache_path)

    return dataset


def load_pickle(path: Path | str, **kwargs: Any) -> Any:
    """
    Load a pickle file.

    Args:
        path: The path to the pickle file.
        **kwargs: Additional arguments to pass to the pickle.loads function.

    Returns:
        The loaded pickle file.
    """
    return pickle.loads(Path(path).read_bytes(), **kwargs)


def dump_pickle(x: Any, path: Path | str, **kwargs: Any) -> None:
    """
    Dump an object into a pickle file.

    Args:
        x: The object to dump.
        path: The path to the pickle file.
        **kwargs: Additional arguments to pass to the pickle.dumps function.
    """
    Path(path).write_bytes(pickle.dumps(x, **kwargs))


# Inspired by: https://github.com/yandex-research/rtdl/blob/a4c93a32b334ef55d2a0559a4407c8306ffeeaee/lib/data.py#L20
# TODO: fix this hideous output type
def normalize(
    X: ArrayDict,
    normalization: Normalization,
    seed: int | None,
    return_normalizer: bool = False,
) -> ArrayDict | tuple[ArrayDict, StandardScaler | MinMaxScaler | QuantileTransformer]:
    """
    Normalize the input data.

    Args:
        X: The data to normalize.
        normalization: The normalization to use. Can be "standard", "minmax", or "quantile".
        seed: The seed to use for the random state. Optional, default is None.
        return_normalizer: Whether to return the normalizer. Optional, default is False.

    Returns:
        The normalized data. If return_normalizer is True, will return a tuple with the
            normalized data and the normalizer.
    """
    X_train = X["train"]
    if normalization == "standard":
        normalizer = StandardScaler()
    elif normalization == "minmax":
        normalizer = MinMaxScaler()
    elif normalization == "quantile":
        normalizer = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(X["train"].shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    normalizer.fit(X_train)
    if return_normalizer:
        return {k: normalizer.transform(v) for k, v in X.items()}, normalizer
    return {k: normalizer.transform(v) for k, v in X.items()}


# TODO: is there any relationship between this function and the cat_process_nans function?
# Can they be made a little more similar to each other (in terms of signature)?
def num_process_nans(dataset: Dataset, policy: NumNanPolicy | None) -> Dataset:
    """
    Process the NaN values in the dataset.

    Args:
        dataset: The dataset to process.
        policy: The policy to use to process the NaN values. Can be "drop-rows" or "mean".
            Optional, default is None.

    Returns:
        The processed dataset.
    """
    assert dataset.X_num is not None
    nan_masks = {k: np.isnan(v) for k, v in dataset.X_num.items()}
    if not any(x.any() for x in nan_masks.values()):
        assert policy is None
        return dataset

    assert policy is not None
    if policy == "drop-rows":
        valid_masks = {k: ~v.any(1) for k, v in nan_masks.items()}
        assert valid_masks["test"].all(), "Cannot drop test rows, since this will affect the final metrics."
        new_data = {}
        for data_name in ["X_num", "X_cat", "y"]:
            data_dict = getattr(dataset, data_name)
            if data_dict is not None:
                new_data[data_name] = {k: v[valid_masks[k]] for k, v in data_dict.items()}
        dataset = replace(dataset, **new_data)  # type: ignore[arg-type]
    elif policy == "mean":
        new_values = np.nanmean(dataset.X_num["train"], axis=0)  # type: ignore[index]
        X_num = deepcopy(dataset.X_num)
        for k, v in X_num.items():  # type: ignore[union-attr]
            num_nan_indices = np.where(nan_masks[k])
            v[num_nan_indices] = np.take(new_values, num_nan_indices[1])
        dataset = replace(dataset, X_num=X_num)
    else:
        raise ValueError(f"Unknown policy: {policy}")
    return dataset


def cat_process_nans(X: ArrayDict, policy: CatNanPolicy | None) -> ArrayDict:
    """
    Process the NaN values in the categorical data.

    Args:
        X: The data to process.
        policy: The policy to use to process the NaN values. Can be "most_frequent".
            Optional, default is None.

    Returns:
        The processed data.
    """
    assert X is not None
    nan_masks = {k: v == CAT_MISSING_VALUE for k, v in X.items()}
    if any(x.any() for x in nan_masks.values()):
        if policy is None:
            X_new = X
        elif policy == "most_frequent":
            imputer = SimpleImputer(missing_values=CAT_MISSING_VALUE, strategy=policy)
            imputer.fit(X["train"])
            X_new = {k: cast(np.ndarray, imputer.transform(v)) for k, v in X.items()}
        else:
            raise ValueError(f"Unknown cat_nan_policy: {policy}")
    else:
        assert policy is None
        X_new = X
    return X_new


def cat_drop_rare(X: ArrayDict, min_frequency: float) -> ArrayDict:
    """
    Drop the rare categories in the categorical data.

    Args:
        X: The data to drop the rare categories from.
        min_frequency: The minimum frequency threshold of the categories to keep. Has to be between 0 and 1.

    Returns:
        The processed data.
    """
    assert 0.0 < min_frequency < 1.0, "min_frequency has to be between 0 and 1"
    min_count = round(len(X["train"]) * min_frequency)
    X_new: dict[str, list[Any]] = {x: [] for x in X}
    for column_idx in range(X["train"].shape[1]):
        counter = Counter(X["train"][:, column_idx].tolist())
        popular_categories = {k for k, v in counter.items() if v >= min_count}
        for part, _ in X_new.items():
            X_new[part].append(
                [(x if x in popular_categories else CAT_RARE_VALUE) for x in X[part][:, column_idx].tolist()]
            )
    return {k: np.array(v).T for k, v in X_new.items()}


def cat_encode(
    X: ArrayDict,
    encoding: CatEncoding | None,  # TODO: add "ordinal" as one of the options, maybe?
    y_train: np.ndarray | None,
    seed: int | None,
    return_encoder: bool = False,
) -> tuple[ArrayDict, bool, Any | None]:
    """
    Encode the categorical data.

    Args:
        X: The data to encode.
        encoding: The encoding to use. Can be "one-hot" or "counter". Default is None.
            If None, will use the "ordinal" encoding.
        y_train: The target values. Optional, default is None. Will only be used for the "counter" encoding.
        seed: The seed to use for the random state. Optional, default is None.
        return_encoder: Whether to return the encoder. Optional, default is False.

    Returns:
        A tuple with the following values:
            - The encoded data.
            - A boolean value indicating if the data was converted to numerical.
            - The encoder, if return_encoder is True. None otherwise.
    """
    if encoding != "counter":
        y_train = None

    # Step 1. Map strings to 0-based ranges

    if encoding is None:
        unknown_value = np.iinfo("int64").max - 3
        oe = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=unknown_value,
            dtype="int64",
        ).fit(X["train"])
        encoder = make_pipeline(oe)
        encoder.fit(X["train"])
        X = {k: encoder.transform(v) for k, v in X.items()}
        max_values = X["train"].max(axis=0)
        for part in X:
            if part == "train":
                continue
            for column_idx in range(X[part].shape[1]):
                X[part][X[part][:, column_idx] == unknown_value, column_idx] = max_values[column_idx] + 1
        if return_encoder:
            return X, False, encoder
        return X, False, None

    # Step 2. Encode.

    if encoding == "one-hot":
        ohe = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            dtype=np.float32,
        )
        encoder = make_pipeline(ohe)

        # encoder.steps.append(('ohe', ohe))
        encoder.fit(X["train"])
        X = {k: encoder.transform(v) for k, v in X.items()}
    elif encoding == "counter":
        assert y_train is not None
        assert seed is not None
        loe = LeaveOneOutEncoder(sigma=0.1, random_state=seed, return_df=False)
        encoder.steps.append(("loe", loe))
        encoder.fit(X["train"], y_train)
        X = {k: encoder.transform(v).astype("float32") for k, v in X.items()}
        if not isinstance(X["train"], pd.DataFrame):
            X = {k: v.values for k, v in X.items()}  # type: ignore[attr-defined]
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    if return_encoder:
        return X, True, encoder
    return X, True, None


def build_target(y: ArrayDict, policy: YPolicy | None, task_type: TaskType) -> tuple[ArrayDict, dict[str, Any]]:
    """
    Build the target and return the target values metadata.

    Args:
        y: The target values.
        policy: The policy to use to build the target. Can be "default". Optional, default is None.
            If none, it will no-op.
        task_type: The type of the task.

    Returns:
        A tuple with the target values and the target values metadata.
    """
    info: dict[str, Any] = {"policy": policy}
    if policy is None:
        pass
    elif policy == "default":
        if task_type == TaskType.REGRESSION:
            mean, std = float(y["train"].mean()), float(y["train"].std())
            y = {k: (v - mean) / std for k, v in y.items()}
            info["mean"] = mean
            info["std"] = std
    else:
        raise ValueError(f"Unknown policy: {policy}")
    return y, info
