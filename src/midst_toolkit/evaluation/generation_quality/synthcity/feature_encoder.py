from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)


def validate_shape(x: np.ndarray, n_dim: int) -> np.ndarray:
    """
    Perform validation of the shape of x against the specified ``n_dim``.

    Args:
        x: Numpy array to be validated
        n_dim: value that determines the validation

    Raises:
        ValueError: Thrown when there is a dissonance between the ``n_dim`` and data shape of x)

    Returns:
        Returns the data, as long as it has the right shape.
    """
    if n_dim == 1:
        if x.ndim == 2:
            x = np.squeeze(x, axis=1)
        if x.ndim != 1:
            raise ValueError("array must be 1D")
        return x
    if n_dim == 2:
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("array must be 2D")
        return x
    raise ValueError("n_dim must be 1 or 2")


class FeatureEncoder(TransformerMixin, BaseEstimator):  # type: ignore
    def __init__(self, n_dim_in: int = 1, n_dim_out: int = 2) -> None:
        """
        Base feature encoder with sklearn-style API.

        Args:
            n_dim_in: Size of the input to the feature encoder. Defaults to 1.
            n_dim_out: Size of the output from the feature encoder. Defaults to 2.
        """
        super().__init__()
        self.n_dim_in = n_dim_in
        self.n_dim_out = n_dim_out
        self.n_features_out: int
        self.feature_name_in: str
        self.feature_names_out: list[str]
        self.feature_types_out: list[str]
        self.categorical: bool = False

    def fit(self, x: pd.Series, y: Any = None, **kwargs: Any) -> FeatureEncoder:
        """
        Fit the feature encoder using the input Pandas series x and possibly the layers in the form of y.

        Args:
            x: Input for fitting the encoder
            y: Optional labels that might be used in fitting the encoder. Defaults to None.
            kwargs: Other settings to be processed

        Returns:
            The fitted FeatureEncoder object.
        """
        self.feature_name_in = x.name
        self.feature_type_in = self._get_feature_type(x)
        input = validate_shape(x.values, self.n_dim_in)
        output = self._fit(input, **kwargs)._transform(input)
        self._out_shape = (-1, *output.shape[1:])  # for inverse_transform
        output = validate_shape(output, self.n_dim_out)
        if self.n_dim_out == 1:
            self.n_features_out = 1
        else:
            self.n_features_out = output.shape[1]
        self.feature_names_out = self.get_feature_names_out()
        self.feature_types_out = self.get_feature_types_out(output)
        return self

    def _fit(self, x: np.ndarray, **kwargs: Any) -> FeatureEncoder:
        return self

    def transform(self, x: pd.Series) -> pd.DataFrame | pd.Series:
        """
        Take in the Series and use the encoder to transform the series after fitting.

        Args:
            x: The input to be transformed.

        Returns:
            The transformed input.
        """
        data = validate_shape(x.values, self.n_dim_in)
        out = self._transform(data)
        out = validate_shape(out, self.n_dim_out)
        if self.n_dim_out == 1:
            return pd.Series(out, name=self.feature_name_in)
        return pd.DataFrame(out, columns=self.feature_names_out)

    def _transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def get_feature_names_out(self) -> list[str]:
        """A list of the names of the features being encoded."""
        n = self.n_features_out
        if n == 1:
            return [self.feature_name_in]
        return [f"{self.feature_name_in}_{i}" for i in range(n)]

    def get_feature_types_out(self, output: np.ndarray) -> list[str]:
        """A list of the name of the features produced by the encoder."""
        t = self._get_feature_type(output)
        return [t] * self.n_features_out

    def _get_feature_type(self, x: Any) -> str:
        """A string indicating the feature type associated with the input."""
        if self.categorical:
            return "discrete"
        if np.issubdtype(x.dtype, np.floating):
            return "continuous"
        if np.issubdtype(x.dtype, np.datetime64):
            return "datetime"
        return "discrete"

    def inverse_transform(self, df: pd.DataFrame | pd.Series) -> pd.Series:
        """Reverse the encoder mapping."""
        y = df.values.reshape(self._out_shape)
        x = self._inverse_transform(y)
        x = validate_shape(x, 1)
        return pd.Series(x, name=self.feature_name_in)

    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data

    @classmethod
    def wraps(cls: type, encoder_class: TransformerMixin, **params: Any) -> type[FeatureEncoder]:
        """Wraps sklearn transformer to FeatureEncoder."""

        class WrappedEncoder(FeatureEncoder):
            def __init__(self, n_dim_in: int = 2, *args: Any, **kwargs: Any) -> None:
                self.encoder = encoder_class(n_dim_in, *args, **kwargs)

            def _fit(self, x: np.ndarray, **kwargs: Any) -> FeatureEncoder:
                self.encoder.fit(x, **kwargs)
                return self

            def _transform(self, x: np.ndarray) -> np.ndarray:
                return self.encoder.transform(x)

            def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
                return self.encoder.inverse_transform(data)

            def get_feature_names_out(self) -> list[str]:
                return list(self.encoder.get_feature_names_out([self.feature_name_in]))

        for attr in ("__name__", "__qualname__", "__doc__"):
            setattr(WrappedEncoder, attr, getattr(encoder_class, attr))
        for attr, val in params.items():
            setattr(WrappedEncoder, attr, val)

        return WrappedEncoder


OneHotEncoder = FeatureEncoder.wraps(OneHotEncoder, categorical=True, handle_unknown="ignore")
OrdinalEncoder = FeatureEncoder.wraps(OrdinalEncoder, categorical=True)
LabelEncoder = FeatureEncoder.wraps(LabelEncoder, n_dim_out=1, categorical=True)
StandardScaler = FeatureEncoder.wraps(StandardScaler)
MinMaxScaler = FeatureEncoder.wraps(MinMaxScaler)
RobustScaler = FeatureEncoder.wraps(RobustScaler)


class DatetimeEncoder(FeatureEncoder):
    """Datetime variables encoder."""

    def __init__(self, n_dim_in: int = 1, n_dim_out: int = 1):
        """
        Datetime variables encoder.

        Args:
            n_dim_in: Size of the input to the feature encoder. Defaults to 1.
            n_dim_out: Size of the output from the feature encoder. Defaults to 1.
        """
        super().__init__(n_dim_in, n_dim_out)

    def _transform(self, x: np.ndarray) -> np.ndarray:
        return pd.to_numeric(x).astype(float)

    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return pd.to_datetime(data)
