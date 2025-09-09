"""
Code is heavily inspired by SynthEvals approach to preprocessing
https://github.com/schneiderkamplab/syntheval/tree/main/src/syntheval/utils.
"""

from logging import INFO

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from midst_toolkit.common.logger import log


def is_none_or_empty(list_to_check: list[str] | None) -> bool:
    """
    Function to check with the provided argument is None or is an empty list.

    Args:
        list_to_check: List to check.

    Returns:
        True if the list provided is empty or the variable is None
    """
    return list_to_check is None or len(list_to_check) == 0


class SynthEvalDataframeEncoding:
    def __init__(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        categorical_columns: list[str] | None,
        numerical_columns: list[str] | None,
        holdout_data: pd.DataFrame = None,
    ) -> None:
        """
        A class responsible for fitting encoders and scalers for categorical and numerical columns of dataframes,
        respectively, These transformations are fitted against the concatenation of the real, synthetic, and any held
        out data that is provided to this class. The categorical and numerical columns lists should correspond to
        the column names for each of the dataframes provided. When using encode and decode, these columns must also
        be present in the provided dataframes to be transformed. Any extant columns in the dataframe not specified
        in these lists will be unmodified. Currently, categorical values are transformed using an Ordinal Encoder and
        numerical values are scaled using a MinMax Scaler.

        NOTE: Upon initialization, the transformations are only fitted against the provided dataframes. These
        dataframes are NOT transformed.

        Args:
            real_data: Dataframe of real data. To be combined with synthetic data to fit the transforms.
            synthetic_data: Dataframe of synthetic data. To be combined with real data to fit the transforms.
            categorical_columns: Column names associated with the categorical variables to be used to fit the
                Ordinal Encoder.
            numerical_columns: Column names associated with the numerical variables to be used to fit the
                MinMax Scaler.
            holdout_data: Any holdout or otherwise auxiliary dataframe to be included as part of the transformation
                fitting process. If none, only the real and synthetic dataframes are used. Defaults to None.
        """
        assert not is_none_or_empty(categorical_columns) or not is_none_or_empty(numerical_columns), (
            "Either categorical or numerical columns must be provided."
        )

        joint_dataframe = pd.concat((real_data.reset_index(), synthetic_data.reset_index()), axis=0)
        if holdout_data is not None:
            joint_dataframe = pd.concat((joint_dataframe.reset_index(), holdout_data.reset_index()), axis=0)

        self.categorical_columns = None
        if not is_none_or_empty(categorical_columns):
            self.categorical_columns = categorical_columns
            self.ordinal_encoder = OrdinalEncoder().fit(joint_dataframe[self.categorical_columns])

        self.numerical_columns = None
        if not is_none_or_empty(numerical_columns):
            self.numerical_columns = numerical_columns
            self.numerical_encoder = MinMaxScaler().fit(joint_dataframe[self.numerical_columns])

    def encode(self, data_to_encode: pd.DataFrame) -> pd.DataFrame:
        """
        Use the fitted categorical and numerical column transformations to encode/scale the data in the provided
        dataframe. This assumes that the ``data_to_encode`` shares the same columns used to fit the respective
        transforms in ``self.categorical_columns`` and ``self.numerical_columns``.

        NOTE: This is an immutable function. It is NOT an in-place operation.

        Args:
            data_to_encode: Dataframe to transform with the fitted transformations.

        Returns:
            New dataframe with the columns encoded/scaled.
        """
        # Deep copy the dataframe
        encoded_data = data_to_encode.copy()
        if self.categorical_columns is not None:
            encoded_data[self.categorical_columns] = self.ordinal_encoder.transform(
                encoded_data[self.categorical_columns]
            ).astype("int")
        if self.numerical_columns is not None:
            encoded_data[self.numerical_columns] = self.numerical_encoder.transform(
                encoded_data[self.numerical_columns]
            )
        return encoded_data

    def decode(self, data_to_decode: pd.DataFrame) -> pd.DataFrame:
        """
        Assumes that the provided dataframe has been previously transformed by this class or has the appropriate
        columns and values to facilitate inverting the transformation. That is, for example, taking a categorical
        value of 1 and mapping it back to "Cat," the original categorical. This assumes that the ``data_to_decode``
        shares  the same columns used to fit the respective transforms in ``self.categorical_columns`` and
        ``self.numerical_columns``.

        NOTE: This is an immutable function. It is NOT an in-place operation.

        Args:
            data_to_decode: Dataframe containing columns that need to be mapped back to their "original" values.

        Returns:
            Dataframe with the inverse mapping/scaling applied to the specified columns.
        """
        encoded_data = data_to_decode.copy()
        if self.categorical_columns is not None:
            encoded_data[self.categorical_columns] = self.ordinal_encoder.inverse_transform(
                encoded_data[self.categorical_columns]
            )
        if self.numerical_columns is not None:
            encoded_data[self.numerical_columns] = self.numerical_encoder.inverse_transform(
                encoded_data[self.numerical_columns]
            )
        return encoded_data


def infer_categorical_and_numerical_columns(
    dataframe: pd.DataFrame, categorical_threshold: int = 10
) -> tuple[list[str], list[str]]:
    """
    Helper function to take in a dataframe and extract the names of the categorical and numerical columns from the
    dataframe in separate lists. These are used to separately treat columns with these distinct types in downstream
    processing.

    NOTE: It is assumed that after identifying categorical columns, the remaining columns represent NUMERICAL values.

    Args:
        dataframe: Dataframe from which to extract the set of categorical and numerical columns.
        categorical_threshold: Threshold below which a column with numerical values (integer or float for example) is
            deemed to represent a categorical encoding. The threshold is compared to the number of unique values
            present for the column in question. If set to 0 (or below), then no columns with numerical entries will be
            treated as categorical. Defaults to 10.

    Returns:
        A tuple of column names that are deemed as holding categorical values and numerical values, respectively.
    """
    categorical_columns = get_categorical_columns(dataframe, threshold=categorical_threshold)
    log(INFO, f"Automatically extracted categorical columns: {categorical_columns}")

    numerical_columns = [column for column in dataframe.columns if column not in categorical_columns]
    log(INFO, f"Numerical columns inferred as the complement of the categorical columns: {numerical_columns}")

    return categorical_columns, numerical_columns


def get_categorical_columns(dataframe: pd.DataFrame, threshold: int) -> list[str]:
    """
    This is a helper function to identify categorical columns in a dataframe. It is a bit brittle and certainly will
    not cover all possible setups. However, it can be helpful. The threshold variable controls how one deems a column
    with numerical values as constituting a categorical indicator. If a column has threshold unique values or less
    it is deemed a categorical column. For example, a hurricane might be rated from 1 to 5 in an integer based column.
    With a threshold of 10, this column would be added to the set of categorical columns.

    Args:
        dataframe: Dataframe from which to extract column names corresponding to categorical variables.
        threshold: Threshold below which a column with numerical values (integer or float for example) is deemed to
            represent a categorical encoding. The threshold is compared to the number of unique values present for the
            column in question. If set to 0 (or below), then no columns with numerical entries will be treated as
            categorical.

    Returns:
        A list of column names from the provided dataframe that correspond to categorical data.
    """
    categorical_variables: list[str] = []

    for column_name in dataframe.columns:
        # If dtype is an object (as str columns are), assume categorical
        if (
            dataframe[column_name].dtype == "object"
            or is_column_type_numerical(dataframe, column_name)
            and dataframe[column_name].nunique() <= threshold
        ):
            categorical_variables.append(column_name)

    return categorical_variables


def is_column_type_numerical(dataframe: pd.DataFrame, column_name: str) -> bool:
    """
    Determine with a column, as specified by ``column_name`` in the dataframe contains "numerical" values. This is
    a heuristic test based on the discussion in the link below.

    https://stackoverflow.com/questions/37726830/how-to-determine-if-a-number-is-any-type-of-int-core-or-numpy-signed-or-not

    Args:
        dataframe: Dataframe whose column values are to be analyzed as being numerical or not.
        column_name: Name of the column in the dataframe to be considered.

    Returns:
        True if the column contains numerical values. False otherwise.
    """
    column_dtype = dataframe[column_name].dtype

    return np.issubdtype(column_dtype, np.integer) or np.issubdtype(column_dtype, np.floating)
