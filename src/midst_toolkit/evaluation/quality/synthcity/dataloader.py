from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader(metaclass=ABCMeta):
    def __init__(
        self,
        data_type: str,
        data: Any,
        static_features: list[str] | None = None,
        temporal_features: list[str] | None = None,
        sensitive_features: list[str] | None = None,
        important_features: list[str] | None = None,
        outcome_features: list[str] | None = None,
        train_size: float = 0.8,
        random_state: int = 0,
    ) -> None:
        """
        Base class for all data loaders.

        Args:
            data_type: The type of DataLoader, currently supports "generic"
            data: The object that contains the data
            static_features: List of feature names that are static features (as opposed to temporal features).
                Defaults to None.
            temporal_features: List of feature names that are temporal features, i.e. observed over time.
                Defaults to None.
            sensitive_features: Name of sensitive features. Defaults to None.
            important_features: Only relevant for SurvivalGAN method. Defaults to None.
            outcome_features: The feature name that provides labels for downstream tasks. Defaults to None.
            train_size: Proportion of data to be used for training, versus evaluation. Defaults to 0.8.
            random_state: Random state for sampling from the dataloaders. Defaults to 0.
        """
        self.static_features = static_features if static_features else []
        self.temporal_features = temporal_features if temporal_features else []
        self.sensitive_features = sensitive_features if sensitive_features else []
        self.important_features = important_features if important_features else []
        self.outcome_features = outcome_features if outcome_features else []
        self.random_state = random_state

        self.data = data
        self.data_type = data_type
        self.train_size = train_size

    def raw(self) -> Any:
        """Just return the data in the dataloader."""
        return self.data

    @abstractmethod
    def unpack(self, as_numpy: bool = False, pad: bool = False) -> Any:
        """A method that unpacks the columns and returns features and labels (X, y)."""
        ...

    @abstractmethod
    def decorate(self, data: Any) -> DataLoader:
        """
        A method that creates a new instance of DataLoader by decorating the input data with the same DataLoader
        properties (e.g. sensitive features, target column, etc.).
        """
        ...

    def type(self) -> str:
        """Return data type."""
        return self.data_type

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """Return shape of the data."""
        ...

    @property
    @abstractmethod
    def columns(self) -> list:
        """Return list of data columns."""
        ...

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """A method that returns the pandas dataframe that contains all features and samples."""
        ...

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        """A method that returns the numpy array that contains all features and samples."""
        ...

    @property
    def values(self) -> np.ndarray:
        """Pass through to the numpy method."""
        return self.to_numpy()

    @abstractmethod
    def info(self) -> dict:
        """A method that returns a dictionary of DataLoader information."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """A method that returns the number of samples in the DataLoader."""
        ...

    @staticmethod
    @abstractmethod
    def from_info(data: pd.DataFrame, info: dict) -> DataLoader:
        """A static method that creates a DataLoader from the data and the information dictionary."""
        ...

    @abstractmethod
    def sample(self, count: int, random_state: int = 0) -> DataLoader:
        """Returns a new DataLoader that contains a random subset of N samples."""
        ...

    @abstractmethod
    def drop(self, columns: list | None) -> DataLoader:
        """Returns a new DataLoader with a list of columns dropped."""
        ...

    @abstractmethod
    def __getitem__(self, feature: str | list) -> Any:
        """Get an item as specified in the feature argument."""
        ...

    @abstractmethod
    def __setitem__(self, feature: str, val: Any) -> None:
        """Set an item in the dataloader to the provided value."""
        ...

    @abstractmethod
    def train(self) -> DataLoader:
        """Returns a DataLoader containing the training set."""
        ...

    @abstractmethod
    def test(self) -> DataLoader:
        """Returns a DataLoader containing the test set."""
        ...

    def __repr__(self, *args: Any, **kwargs: Any) -> str:
        """Return a string representation."""
        return self.to_dataframe().__repr__(*args, **kwargs)

    def _repr_html_(self, *args: Any, **kwargs: Any) -> Any:
        """Return a string representation in html format."""
        return self.to_dataframe().to_html(*args, **kwargs)

    @abstractmethod
    def fillna(self, value: Any) -> DataLoader:
        """Returns a DataLoader with NaN filled by the provided number(s)."""
        ...

    @abstractmethod
    def compression_protected_features(self) -> list:
        """No idea."""
        ...

    def domain(self) -> str | None:
        """Domain of the data."""
        return None

    @abstractmethod
    def is_tabular(self) -> bool:
        """Specifies whether the dataloader represents tabular data."""
        ...

    @abstractmethod
    def get_fairness_column(self) -> str | Any:
        """Get the name of the column associated with Fairness."""
        ...


class GenericDataLoader(DataLoader):
    def __init__(
        self,
        data: pd.DataFrame | list | np.ndarray,
        sensitive_features: list[str] | None = None,
        important_features: list[str] | None = None,
        target_column: str | None = None,
        fairness_column: str | None = None,
        domain_column: str | None = None,
        random_state: int = 0,
        train_size: float = 0.8,
        **kwargs: Any,
    ) -> None:
        """
        Data loader for generic tabular data.

        Args:
            data: The dataset. Either a Pandas DataFrame, list, or a Numpy Array.
            sensitive_features:  Name of sensitive features.. Defaults to None.
            important_features: Only relevant for SurvivalGAN method. Defaults to None.
            target_column: The feature name that provides labels for downstream tasks. Defaults to None.
            fairness_column:  Optional fairness column label, used for fairness benchmarking. Defaults to None.
            domain_column: Optional domain label, used for domain adaptation algorithms. Defaults to None.
            random_state: Random state for sampling from the dataloaders. Defaults to 0.
            train_size: Proportion of data to be used for training, versus evaluation. Defaults to 0.8.
            kwargs: Other settings to be processed.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        data.columns = data.columns.astype(str)
        if target_column is not None:
            self.target_column = target_column
        elif len(data.columns) > 0:
            self.target_column = data.columns[-1]
        else:
            self.target_column = "---"

        self.fairness_column = fairness_column
        self.domain_column = domain_column

        super().__init__(
            data_type="generic",
            data=data,
            static_features=list(data.columns),
            sensitive_features=sensitive_features,
            important_features=important_features,
            outcome_features=[self.target_column],
            random_state=random_state,
            train_size=train_size,
            **kwargs,
        )

    @property
    def shape(self) -> tuple:
        """Return the shape of the data."""
        return self.data.shape

    def domain(self) -> str | None:
        """Return the domain column if it exists."""
        return self.domain_column

    def get_fairness_column(self) -> str | Any:
        """Return the fairness column if it exists."""
        return self.fairness_column

    @property
    def columns(self) -> list:
        """Return a list of the data columns."""
        return list(self.data.columns)

    def compression_protected_features(self) -> list:
        """No idea."""
        out = [self.target_column]
        domain = self.domain()

        if domain is not None:
            out.append(domain)

        return out

    def unpack(self, as_numpy: bool = False, pad: bool = False) -> Any:
        """A method that unpacks the columns and returns features and labels (X, y)."""
        x = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        if as_numpy:
            return np.asarray(x), np.asarray(y)
        return x, y

    def to_dataframe(self) -> pd.DataFrame:
        """A method that returns the pandas dataframe that contains all features and samples."""
        return self.data

    def to_numpy(self) -> np.ndarray:
        """A method that returns the numpy array that contains all features and samples."""
        return self.to_dataframe().to_numpy()

    def info(self) -> dict:
        """A method that returns a dictionary of DataLoader information."""
        return {
            "data_type": self.data_type,
            "len": len(self),
            "static_features": self.static_features,
            "sensitive_features": self.sensitive_features,
            "important_features": self.important_features,
            "outcome_features": self.outcome_features,
            "target_column": self.target_column,
            "fairness_column": self.fairness_column,
            "domain_column": self.domain_column,
            "train_size": self.train_size,
        }

    def __len__(self) -> int:
        """A method that returns the number of samples in the DataLoader."""
        return len(self.data)

    def decorate(self, data: Any) -> DataLoader:
        """
        A method that creates a new instance of DataLoader by decorating the input data with the same DataLoader
        properties (e.g. sensitive features, target column, etc.).
        """
        return GenericDataLoader(
            data,
            sensitive_features=self.sensitive_features,
            important_features=self.important_features,
            target_column=self.target_column,
            random_state=self.random_state,
            train_size=self.train_size,
            fairness_column=self.fairness_column,
            domain_column=self.domain_column,
        )

    def sample(self, count: int, random_state: int = 0) -> DataLoader:
        """Returns a new DataLoader that contains a random subset of N samples."""
        return self.decorate(self.data.sample(count, random_state=random_state))

    def drop(self, columns: list | None = None) -> DataLoader:
        """Returns a new DataLoader with a list of columns dropped."""
        return self.decorate(self.data.drop(columns=(columns if columns else [])))

    @staticmethod
    def from_info(data: pd.DataFrame, info: dict) -> GenericDataLoader:
        """A static method that creates a DataLoader from the data and the information dictionary."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Invalid data type {type(data)}")

        return GenericDataLoader(
            data,
            sensitive_features=info["sensitive_features"],
            important_features=info["important_features"],
            target_column=info["target_column"],
            fairness_column=info["fairness_column"],
            domain_column=info["domain_column"],
            train_size=info["train_size"],
        )

    def __getitem__(self, feature: str | list | int) -> Any:
        """Get an item from the dataloader."""
        return self.data[feature]

    def __setitem__(self, feature: str, val: Any) -> None:
        """Replace an item in the dataloader with the specified value."""
        self.data[feature] = val

    def _train_test_split(self) -> tuple:
        """Split the dataset into train and test sets according to a specified ratio."""
        stratify = None
        if self.target_column in self.data:
            target = self.data[self.target_column]
            if target.value_counts().min() > 1:
                stratify = target

        return train_test_split(
            self.data,
            train_size=self.train_size,
            random_state=self.random_state,
            stratify=stratify,
        )

    def train(self) -> DataLoader:
        """Returns a DataLoader containing the training set."""
        train_data, _ = self._train_test_split()
        return self.decorate(train_data.reset_index(drop=True))

    def test(self) -> DataLoader:
        """Returns a DataLoader containing the training set."""
        _, test_data = self._train_test_split()
        return self.decorate(test_data.reset_index(drop=True))

    def fillna(self, value: Any) -> DataLoader:
        """Returns a DataLoader with NaN filled by the provided number(s)."""
        self.data = self.data.fillna(value)
        return self

    def is_tabular(self) -> bool:
        """Always represents a tabular dataset."""
        return True


def create_from_info(data: pd.DataFrame, info: dict) -> DataLoader:
    """Helper for creating a DataLoader from existing information."""
    if info["data_type"] == "generic":
        return GenericDataLoader.from_info(data, info)
    raise RuntimeError(f"invalid datatype {info}")
