import hashlib
import json
import math
import pickle
from collections import Counter
from collections.abc import Callable, Generator, Iterator
from copy import deepcopy
from dataclasses import astuple, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Self, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ruff: noqa: N812
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
from torch import Tensor, nn

from midst_toolkit.common.enumerations import PredictionType, TaskType


Normalization = Literal["standard", "quantile", "minmax"]
NumNanPolicy = Literal["drop-rows", "mean"]
CatNanPolicy = Literal["most_frequent"]
CatEncoding = Literal["one-hot", "counter"]
YPolicy = Literal["default"]


ArrayDict = dict[str, np.ndarray]
ModuleType = str | Callable[..., nn.Module]

CAT_MISSING_VALUE = "__nan__"
CAT_RARE_VALUE = "__rare__"


@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Normalization | None = None
    num_nan_policy: NumNanPolicy | None = None
    cat_nan_policy: CatNanPolicy | None = None
    cat_min_frequency: float | None = None
    cat_encoding: CatEncoding | None = None
    y_policy: YPolicy | None = "default"


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
    def from_dir(cls, dir_: Path | str) -> "Dataset":
        dir_ = Path(dir_)
        splits = [k for k in ["train", "val", "test"] if dir_.joinpath(f"y_{k}.npy").exists()]

        def load(item: str) -> ArrayDict:
            return {x: cast(np.ndarray, np.load(dir_ / f"{item}_{x}.npy", allow_pickle=True)) for x in splits}

        if Path(dir_ / "info.json").exists():
            info = json.loads(Path(dir_ / "info.json").read_text())
        else:
            info = None
        # ruff: noqa: SIM108

        return Dataset(
            load("X_num") if dir_.joinpath("X_num_train.npy").exists() else None,
            load("X_cat") if dir_.joinpath("X_cat_train.npy").exists() else None,
            load("y"),
            {},
            TaskType(info["task_type"]),
            info.get("n_classes"),
        )

    @property
    def is_binclass(self) -> bool:
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num["train"].shape[1]

    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat["train"].shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, part: str | None) -> int:
        return sum(map(len, self.y.values())) if part is None else len(self.y[part])

    @property
    def nn_output_dim(self) -> int:
        if self.is_multiclass:
            assert self.n_classes is not None
            return self.n_classes
        return 1

    def get_category_sizes(self, part: str) -> list[int]:
        return [] if self.X_cat is None else get_category_sizes(self.X_cat[part])

    def calculate_metrics(
        self,
        predictions: dict[str, np.ndarray],
        prediction_type: str | None,
    ) -> dict[str, Any]:
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


def get_category_sizes(X: torch.Tensor | np.ndarray) -> list[int]:
    XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
    return [len(set(x)) for x in XT]


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str | TaskType,
    prediction_type: str | PredictionType | None,
    y_info: dict[str, Any],
) -> dict[str, Any]:
    # Example: calculate_metrics(y_true, y_pred, 'binclass', 'logits', {})
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
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    if std is not None:
        rmse *= std
    return rmse


def _get_labels_and_probs(
    y_pred: np.ndarray, task_type: TaskType, prediction_type: PredictionType | None
) -> tuple[np.ndarray, np.ndarray | None]:
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


class Classifier(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dim_t: int,
        hidden_sizes: list[int],
        dropout_prob: float = 0.5,
        num_heads: int = 2,
        num_layers: int = 1,
    ):
        super(Classifier, self).__init__()

        self.dim_t = dim_t
        self.proj = nn.Linear(d_in, dim_t)

        self.transformer_layer = nn.Transformer(d_model=dim_t, nhead=num_heads, num_encoder_layers=num_layers)

        self.time_embed = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t))

        # Create a list to hold the layers
        layers: list[nn.Module] = []

        # Add input layer
        layers.append(nn.Linear(dim_t, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))  # Batch Normalization
        layers.append(nn.Dropout(p=dropout_prob))

        # Add hidden layers with batch normalization and different activation
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.LeakyReLU())  # Different activation
            layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))  # Batch Normalization
            layers.append(nn.Dropout(p=dropout_prob))

        # Add output layer
        layers.append(nn.Linear(hidden_sizes[-1], d_out))

        # Create a Sequential model from the list of layers
        self.model = nn.Sequential(*layers)

    def forward(self, x, timesteps):
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        x = self.proj(x) + emb
        # x = self.transformer_layer(x, x)
        return self.model(x)


def get_table_info(df: pd.DataFrame, domain_dict: dict[str, Any], y_col: str) -> dict[str, Any]:
    cat_cols = []
    num_cols = []
    for col in df.columns:
        if col in domain_dict and col != y_col:
            if domain_dict[col]["type"] == "discrete":
                cat_cols.append(col)
            else:
                num_cols.append(col)

    df_info: dict[str, Any] = {}
    df_info["cat_cols"] = cat_cols
    df_info["num_cols"] = num_cols
    df_info["y_col"] = y_col
    df_info["n_classes"] = 0
    df_info["task_type"] = "multiclass"

    return df_info


def get_model_params(rtdl_params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "num_classes": 0,
        "is_y_cond": "none",
        "rtdl_params": {"d_layers": [512, 1024, 1024, 1024, 1024, 512], "dropout": 0.0}
        if rtdl_params is None
        else rtdl_params,
    }


def get_T_dict() -> dict[str, Any]:
    # ruff: noqa: N802
    return {
        "seed": 0,
        "normalization": "quantile",
        "num_nan_policy": None,
        "cat_nan_policy": None,
        "cat_min_frequency": None,
        "cat_encoding": None,
        "y_policy": "default",
    }


def make_dataset_from_df(
    # ruff: noqa: PLR0915, PLR0912
    df: pd.DataFrame,
    T: Transformations,
    is_y_cond: str,
    df_info: pd.DataFrame,
    ratios: list[float] | None = None,
    std: float = 0,
) -> tuple[Dataset, dict[int, LabelEncoder], list[int]]:
    """
    The order of the generated dataset: (y, X_num, X_cat).

    is_y_cond:
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

    Note: For now, n_classes has to be set to 0. This is because our matrix is the concatenation
    of (X_num, X_cat). In this case, if we have is_y_cond == 'concat', we can guarantee that y
    is the first column of the matrix.
    However, if we have n_classes > 0, then y is not the first column of the matrix.
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
        X_cat: dict[str, np.ndarray] | None = {} if df_info["cat_cols"] is not None or is_y_cond == "concat" else None
        X_num: dict[str, np.ndarray] | None = {} if df_info["num_cols"] is not None else None
        y = {}

        cat_cols_with_y = []
        if df_info["cat_cols"] is not None:
            cat_cols_with_y += df_info["cat_cols"]
        if is_y_cond == "concat":
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
        X_num = {} if df_info["num_cols"] is not None or is_y_cond == "concat" else None
        y = {}

        num_cols_with_y = []
        if df_info["num_cols"] is not None:
            num_cols_with_y += df_info["num_cols"]
        if is_y_cond == "concat":
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

    return transform_dataset(D, T, None), label_encoders, column_orders


def prepare_fast_dataloader(
    D: Dataset,
    # ruff: noqa: N803
    split: str,
    batch_size: int,
    y_type: str = "float",
) -> Generator[tuple[Tensor, ...]]:
    if D.X_cat is not None:
        if D.X_num is not None:
            X = torch.from_numpy(np.concatenate([D.X_num[split], D.X_cat[split]], axis=1)).float()
        else:
            X = torch.from_numpy(D.X_cat[split]).float()
    else:
        assert D.X_num is not None
        X = torch.from_numpy(D.X_num[split]).float()
    y = torch.from_numpy(D.y[split]).float() if y_type == "float" else torch.from_numpy(D.y[split]).long()
    dataloader = FastTensorDataLoader(X, y, batch_size=batch_size, shuffle=(split == "train"))
    while True:
        yield from dataloader


def get_model(
    model_name: str,
    model_params: dict[str, Any],
) -> nn.Module:
    print(model_name)
    if model_name == "mlp":
        return MLPDiffusion(**model_params)
    if model_name == "resnet":
        return ResNetDiffusion(**model_params)

    raise ValueError("Unknown model!")


def update_ema(
    target_params: Iterator[nn.Parameter],
    source_params: Iterator[nn.Parameter],
    rate: float = 0.999,
) -> None:
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)


def transform_dataset(
    dataset: Dataset,
    transformations: Transformations,
    cache_dir: Path | None,
    transform_cols_num: int = 0,
) -> Dataset:
    # WARNING: the order of transformations matters. Moreover, the current
    # implementation is not ideal in that sense.
    if cache_dir is not None:
        transformations_md5 = hashlib.md5(str(transformations).encode("utf-8")).hexdigest()
        transformations_str = "__".join(map(str, astuple(transformations)))
        cache_path = cache_dir / f"cache__{transformations_str}__{transformations_md5}.pickle"
        if cache_path.exists():
            cache_transformations, value = load_pickle(cache_path)
            if transformations == cache_transformations:
                print(f"Using cached features: {cache_dir.name + '/' + cache_path.name}")
                return value
            raise RuntimeError(f"Hash collision for {cache_path}")
    else:
        cache_path = None

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
    # if return_transforms:
    # return dataset, num_transform, cat_transform
    return dataset


def load_pickle(path: Path | str, **kwargs: Any) -> Any:
    # ruff: noqa: D103
    return pickle.loads(Path(path).read_bytes(), **kwargs)


def dump_pickle(x: Any, path: Path | str, **kwargs: Any) -> None:
    # ruff: noqa: D103
    Path(path).write_bytes(pickle.dumps(x, **kwargs))


def num_process_nans(dataset: Dataset, policy: NumNanPolicy | None) -> Dataset:
    # ruff: noqa: D103
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


# Inspired by: https://github.com/yandex-research/rtdl/blob/a4c93a32b334ef55d2a0559a4407c8306ffeeaee/lib/data.py#L20
def normalize(
    X: ArrayDict,
    normalization: Normalization,
    seed: int | None,
    return_normalizer: bool = False,
) -> ArrayDict | tuple[ArrayDict, StandardScaler | MinMaxScaler | QuantileTransformer]:
    # ruff: noqa: D103
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
        # noise = 1e-3
        # if noise > 0:
        #     assert seed is not None
        #     stds = np.std(X_train, axis=0, keepdims=True)
        #     noise_std = noise / np.maximum(stds, noise)  # type: ignore[code]
        #     X_train = X_train + noise_std * np.random.default_rng(seed).standard_normal(
        #         X_train.shape
        #     )
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    normalizer.fit(X_train)
    if return_normalizer:
        return {k: normalizer.transform(v) for k, v in X.items()}, normalizer
    return {k: normalizer.transform(v) for k, v in X.items()}


def cat_process_nans(X: ArrayDict, policy: CatNanPolicy | None) -> ArrayDict:
    # ruff: noqa: D103
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
    # ruff: noqa: D103
    assert 0.0 < min_frequency < 1.0
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
    encoding: CatEncoding | None,
    y_train: np.ndarray | None,
    seed: int | None,
    return_encoder: bool = False,
) -> tuple[ArrayDict, bool, Any | None]:  # (X, is_converted_to_numerical)
    # ruff: noqa: D103
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
    # ruff: noqa: D103
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


class FastTensorDataLoader:
    """
    Defines a faster dataloader for PyTorch tensors.

    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors: Tensor, batch_size: int = 32, shuffle: bool = False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        # ruff: noqa: D105
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]  # type: ignore[assignment]
        self.i = 0
        return self

    def __next__(self):
        # ruff: noqa: D105
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        # ruff: noqa: D105
        return self.n_batches


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class MLP(nn.Module):
    """The MLP model used in [gorishniy2021revisiting].

    The following scheme describes the architecture:

    .. code-block:: text

          MLP: (in) -> Block -> ... -> Block -> Linear -> (out)
        Block: (in) -> Linear -> Activation -> Dropout -> (out)

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            module = MLP.make_baseline(x.shape[1], [3, 5], 0.1, 1)
            assert module(x).shape == (len(x), 1)

    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov,
        Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `MLP`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation: ModuleType,
            dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(
        self,
        *,
        d_in: int,
        d_layers: list[int],
        dropouts: float | list[float],
        activation: str | Callable[[], nn.Module],
        d_out: int,
    ) -> None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)
        assert activation not in ["ReGLU", "GEGLU"]

        self.blocks = nn.ModuleList(
            [
                MLP.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    @classmethod
    def make_baseline(
        cls,
        d_in: int,
        d_layers: list[int],
        dropout: float,
        d_out: int,
    ) -> Self:
        """Create a "baseline" `MLP`.

        This variation of MLP was used in [gorishniy2021revisiting]. Features:

        * all linear layers except for the first one and the last one are of the same dimension
        * the dropout rate is the same for all dropout layers

        Args:
            d_in: the input size
            d_layers: the dimensions of the linear layers. If there are more than two
                layers, then all of them except for the first and the last ones must
                have the same dimension.
            dropout: the dropout rate for all hidden layers
            d_out: the output size
        Returns:
            MLP

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov,
            Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        assert isinstance(dropout, float)
        if len(d_layers) > 2:
            assert len(set(d_layers[1:-1])) == 1, (
                "if d_layers contains more than two elements, then"
                " all elements except for the first and the last ones must be equal."
            )
        return cls(
            d_in=d_in,
            d_layers=d_layers,
            dropouts=dropout,
            activation="ReLU",
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class ResNet(nn.Module):
    """
    The ResNet model used in [gorishniy2021revisiting].

    The following scheme describes the architecture:
    .. code-block:: text
        ResNet: (in) -> Linear -> Block -> ... -> Block -> Head -> (out)
                 |-> Norm -> Linear -> Activation -> Dropout -> Linear -> Dropout ->|
                 |                                                                  |
         Block: (in) ------------------------------------------------------------> Add -> (out)
          Head: (in) -> Norm -> Activation -> Linear -> (out)

    Examples:
        .. testcode::
            x = torch.randn(4, 2)
            module = ResNet.make_baseline(
                d_in=x.shape[1],
                n_blocks=2,
                d_main=3,
                d_hidden=4,
                dropout_first=0.25,
                dropout_second=0.0,
                d_out=1
            )
            assert module(x).shape == (len(x), 1)

    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov,
        Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `ResNet`."""

        def __init__(
            self,
            *,
            d_main: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout_first: float,
            dropout_second: float,
            normalization: ModuleType,
            activation: ModuleType,
            skip_connection: bool,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_main)
            self.linear_first = nn.Linear(d_main, d_hidden, bias_first)
            self.activation = _make_nn_module(activation)
            self.dropout_first = nn.Dropout(dropout_first)
            self.linear_second = nn.Linear(d_hidden, d_main, bias_second)
            self.dropout_second = nn.Dropout(dropout_second)
            self.skip_connection = skip_connection

        def forward(self, x: Tensor) -> Tensor:
            x_input = x
            x = self.normalization(x)
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout_first(x)
            x = self.linear_second(x)
            x = self.dropout_second(x)
            if self.skip_connection:
                x = x_input + x
            return x

    class Head(nn.Module):
        """The final module of `ResNet`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            normalization: ModuleType,
            activation: ModuleType,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            if self.normalization is not None:
                x = self.normalization(x)
            x = self.activation(x)
            return self.linear(x)

    def __init__(
        self,
        *,
        d_in: int,
        n_blocks: int,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
        normalization: ModuleType,
        activation: ModuleType,
        d_out: int,
    ) -> None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()

        self.first_layer = nn.Linear(d_in, d_main)
        if d_main is None:
            d_main = d_in
        self.blocks = nn.Sequential(
            *[
                ResNet.Block(
                    d_main=d_main,
                    d_hidden=d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout_first=dropout_first,
                    dropout_second=dropout_second,
                    normalization=normalization,
                    activation=activation,
                    skip_connection=True,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = ResNet.Head(
            d_in=d_main,
            d_out=d_out,
            bias=True,
            normalization=normalization,
            activation=activation,
        )

    @classmethod
    def make_baseline(
        cls,
        *,
        d_in: int,
        n_blocks: int,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
        d_out: int,
    ) -> Self:
        """
        Create a "baseline" `ResNet`. This variation of ResNet was used in [gorishniy2021revisiting].

        Args:
            d_in: the input size
            n_blocks: the number of Blocks
            d_main: the input size (or, equivalently, the output size) of each Block
            d_hidden: the output size of the first linear layer in each Block
            dropout_first: the dropout rate of the first dropout layer in each Block.
            dropout_second: the dropout rate of the second dropout layer in each Block.
            d_out: Output dimension.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov,
            Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        return cls(
            d_in=d_in,
            n_blocks=n_blocks,
            d_main=d_main,
            d_hidden=d_hidden,
            dropout_first=dropout_first,
            dropout_second=dropout_second,
            normalization="BatchNorm1d",
            activation="ReLU",
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        x = self.first_layer(x)
        x = self.blocks(x)
        return self.head(x)


#### For diffusion


class MLPDiffusion(nn.Module):
    def __init__(
        self,
        d_in: int,
        num_classes: int,
        is_y_cond: str,
        rtdl_params: dict[str, Any],
        dim_t: int = 128,
    ):
        super().__init__()
        self.dim_t = dim_t
        self.num_classes = num_classes
        self.is_y_cond = is_y_cond

        # d0 = rtdl_params['d_layers'][0]

        rtdl_params["d_in"] = dim_t
        rtdl_params["d_out"] = d_in

        self.mlp = MLP.make_baseline(**rtdl_params)

        self.label_emb: nn.Embedding | nn.Linear
        if self.num_classes > 0 and is_y_cond == "embedding":
            self.label_emb = nn.Embedding(self.num_classes, dim_t)
        elif self.num_classes == 0 and is_y_cond == "embedding":
            self.label_emb = nn.Linear(1, dim_t)

        self.proj = nn.Linear(d_in, dim_t)
        self.time_embed = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t))

    def forward(self, x, timesteps, y=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        if self.is_y_cond == "embedding" and y is not None:
            y = y.squeeze() if self.num_classes > 0 else y.resize_(y.size(0), 1).float()
            emb += F.silu(self.label_emb(y))
        x = self.proj(x) + emb
        return self.mlp(x)


class ResNetDiffusion(nn.Module):
    def __init__(
        self,
        d_in: int,
        num_classes: int,
        rtdl_params: dict[str, Any],
        dim_t: int = 256,
        is_y_cond: str | None = None,
    ):
        # ruff: noqa: D107
        super().__init__()
        self.dim_t = dim_t
        self.num_classes = num_classes

        rtdl_params["d_in"] = d_in
        rtdl_params["d_out"] = d_in
        rtdl_params["emb_d"] = dim_t
        self.resnet = ResNet.make_baseline(**rtdl_params)

        self.label_emb: nn.Embedding | nn.Linear
        if self.num_classes > 0 and is_y_cond == "embedding":
            self.label_emb = nn.Embedding(self.num_classes, dim_t)
        elif self.num_classes == 0 and is_y_cond == "embedding":
            self.label_emb = nn.Linear(1, dim_t)

        self.time_embed = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t))

    def forward(self, x, timesteps, y=None):
        # ruff: noqa: D102
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        if y is not None and self.num_classes > 0:
            emb += self.label_emb(y.squeeze())
        return self.resnet(x, emb)


def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].

    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].

    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        # ruff: noqa: D102
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        # ruff: noqa: D102
        return geglu(x)


def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:  # type: ignore[no-untyped-def]
    return (
        (ReGLU() if module_type == "ReGLU" else GEGLU() if module_type == "GEGLU" else getattr(nn, module_type)(*args))
        if isinstance(module_type, str)
        else module_type(*args)
    )
