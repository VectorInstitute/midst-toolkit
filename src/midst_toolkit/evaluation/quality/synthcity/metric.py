from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from midst_toolkit.evaluation.quality.synthcity.dataloader import DataLoader
from midst_toolkit.evaluation.quality.synthcity.one_class import OneClassLayer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetricEvaluator(metaclass=ABCMeta):
    def __init__(
        self,
        reduction: str = "mean",
        n_histogram_bins: int = 10,
        n_folds: int = 3,
        task_type: str = "classification",
        random_state: int = 0,
        workspace: Path = Path("workspace"),
        use_cache: bool = True,
        default_metric: str | None = None,
    ) -> None:
        """
        Base class for all metrics.

        If any method implementation is missing, the class constructor will fail.

        Args:
            reduction: The way to aggregate metrics across folds. Defaults to "mean".
            n_histogram_bins: The number of bins used in histogram calculation. Defaults to 10.
            n_folds: The number of folds in cross validation. Defaults to 3.
            task_type: The type of downstream task.. Defaults to "classification".
            random_state: Random state seed. Defaults to 0.
            workspace: The directory to save intermediate models or results.. Defaults to Path("workspace").
            use_cache:  Whether to use cache. If True, it will try to load saved results in workspace directory
                where possible. Defaults to True.
            default_metric: Type of metric to be used if one not specified. Defaults to None.
        """
        self._reduction = reduction
        self._n_histogram_bins = n_histogram_bins
        self._n_folds = n_folds

        self._task_type = task_type
        self._random_state = random_state
        self._workspace = workspace
        self._use_cache = use_cache
        if default_metric is None:
            default_metric = reduction
        self._default_metric = default_metric

        workspace.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def evaluate(self, x_gt: DataLoader, x_syn: DataLoader) -> dict:
        """Compare two datasets and return a dictionary of metrics."""
        ...

    @abstractmethod
    def evaluate_default(self, x_gt: DataLoader, x_syn: DataLoader) -> float:
        """Default evaluation."""
        ...

    @staticmethod
    @abstractmethod
    def direction() -> str:
        """Direction of metric (bigger better or smaller better)."""
        ...

    @staticmethod
    @abstractmethod
    def type() -> str:
        """Type of metric."""
        ...

    @staticmethod
    @abstractmethod
    def name() -> str:
        """Name of the metric."""
        ...

    @classmethod
    def fqdn(cls) -> str:
        """No idea."""
        return f"{cls.type()}.{cls.name()}"

    def reduction(self) -> Callable:
        """The way in which the input should be reduced if necessary."""
        if self._reduction == "mean":
            return np.mean
        if self._reduction == "max":
            return np.max
        if self._reduction == "min":
            return np.min
        raise ValueError(f"Unknown reduction {self._reduction}")

    def _get_oneclass_model(self, x_gt: np.ndarray) -> OneClassLayer:
        model = OneClassLayer(
            input_dim=x_gt.shape[1],
            rep_dim=x_gt.shape[1],
            center=torch.ones(x_gt.shape[1]) * 10,
        )
        model.fit(torch.from_numpy(x_gt))

        return model.to(DEVICE)

    def _oneclass_predict(self, model: OneClassLayer, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(X).float().to(DEVICE)).cpu().detach().numpy()

    def use_cache(self, path: Path) -> bool:
        """Whether to save information to the provided path."""
        return path.exists() and self._use_cache
