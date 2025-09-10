from abc import ABC, abstractmethod
from logging import INFO
from typing import overload

import pandas as pd

from midst_toolkit.common.logger import log
from midst_toolkit.data_processing.utils import SynthEvalDataframeEncoding


class QualityMetricBase(ABC):
    @abstractmethod
    def compute(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict[str, float]:
        """
        Abstract method for computing a synthetic data quality metric. Should be implemented by inheriting classes
        and return a dictionary of values for the resulting metric computations.

        Args:
            real_data: Real data to which the synthetic data may be compared. In many cases this will be data used
                to TRAIN the model that generated the synthetic data, but not always.
            synthetic_data: Synthetically generated data whose quality is to be assessed.

        Raises:
            NotImplementedError: Must be implemented by inheriting metrics

        Returns:
            a dictionary with string keys with float values computed by the metric. Some metrics return multiple
            statistics. For example, in confidence interval estimation, one might have mean and standard deviation.
        """
        raise NotImplementedError("Inheriting class must define compute")


class SynthEvalQualityMetric(QualityMetricBase, ABC):
    def __init__(
        self,
        categorical_columns: list[str],
        numerical_columns: list[str],
        do_preprocess: bool = False,
    ) -> None:
        """
        Base class for SynthEval metrics. These metrics require designation of the column names that are associated
        with categorical variables and those columns that are associated with numerical variables. These are used
        to facilitate metric computation (for example, when only numerical columns are admissible) and for
        preprocessing if desired. The default preprocessing pipeline for SynthEval is implemented by the
        DataframeEncoding class. If desired this class can preprocess dataframes before performing computation.

        Args:
            categorical_columns: Column names corresponding to the categorical variables of any provided dataframe.
            numerical_columns: Column names corresponding to the numerical variables of any provided dataframe.
            do_preprocess: Whether or not to preprocess the dataframes with the default pipeline used by SynthEval.
                Defaults to False.
        """
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.do_preprocess = do_preprocess

        if do_preprocess:
            log(INFO, "Default preprocessing will be performed during computation.")

    @overload
    def preprocess(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    @overload
    def preprocess(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, holdout_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: ...

    def preprocess(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, holdout_data: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies the default dataframe preprocessing pipeline for SynthEval. This has been pulled into the library
        to allow for this to be optional (it is not in the SynthEval library) and controllable for our purposes.
        The preprocessing classes are fitted to the combination of all the provided dataframes and will be
        refitted any time this is called.

        Args:
            real_data: Real data to which the synthetic data may be compared. In many cases this will be data used
                to TRAIN the model that generated the synthetic data, but not always.
            synthetic_data: Synthetically generated data whose quality is to be assessed.
            holdout_data: An optional set of holdout data. Typically, this will be data drawn from the same
                distribution as ``real_data`` but was explicitly NOT used to train the model that generated
                ``synthetic_data``. Not all metrics will require a holdout set and it is, therefore optional.
                Defaults to None.

        Returns:
            Transformed dataframes for the real, synthetic, and holdout dataframes (if provided)
        """
        log(INFO, "Performing default preprocessing using defined columns.")
        encoder = SynthEvalDataframeEncoding(
            real_data, synthetic_data, self.categorical_columns, self.numerical_columns
        )
        real_data = encoder.encode(real_data)
        synthetic_data = encoder.encode(synthetic_data)
        holdout_data = encoder.encode(holdout_data) if holdout_data else None

        if holdout_data is not None:
            return real_data, synthetic_data, holdout_data
        return real_data, synthetic_data
