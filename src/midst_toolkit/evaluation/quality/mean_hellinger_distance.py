import numpy as np
import pandas as pd

from midst_toolkit.evaluation.metrics_base import SynthEvalQualityMetric


def hellinger_distance(discrete_distribution_1: np.ndarray, discrete_distribution_2: np.ndarray) -> float:
    """
    Compute the empirical Hellinger distance between two discrete probability distributions. Hellinger distance for
    discrete probability distributions $p$ and $q$ is expressed as
    $$\\frac{1}{2} \\cdot \\Vert \\sqrt{p} - \\sqrt{q} \\Vert_2$$.

    Args:
        discrete_distribution_1: First discrete distribution for distance computation
        discrete_distribution_2: Second discrete distribution for distance computation

    Returns:
        Empirical Hellinger distance between the two distributions.
    """
    sum_1 = np.sum(discrete_distribution_1)
    sum_2 = np.sum(discrete_distribution_2)
    assert np.isclose(sum_1, 1.0, atol=1e-4), f"Distribution 1 is not a probability distribution: Sum is {sum_1}"
    assert np.isclose(sum_2, 1.0, atol=1e-4), f"Distribution 2 is not a probability distribution: Sum is {sum_2}"

    sqrt_pdf_1 = np.sqrt(discrete_distribution_1)
    sqrt_pdf_2 = np.sqrt(discrete_distribution_2)
    difference = sqrt_pdf_1 - sqrt_pdf_2
    return 1 / np.sqrt(2) * np.linalg.norm(difference)


class MeanHellingerDistance(SynthEvalQualityMetric):
    def __init__(
        self,
        categorical_columns: list[str],
        numerical_columns: list[str],
        do_preprocess: bool = False,
        include_numerical_columns: bool = True,
    ):
        """
        This class computes the DISCRETE Hellinger Distance between corresponding columns of real and synthetic
        dataframes.

        NOTE: The implementation here is inspired by the SynthEval implementation of the Mean Hellinger Distance
        but fixes a crucial issue. Their way of computing bins for the discrete histograms of numerical values is
        flawed. Here, we make use of the 'auto' binning schemes in numpy to do a better job binning such values into
        histograms

        - For a categorical column, the number of bins for the discrete distributions is established by computing
          the unique values in the column for the REAL DATA. This can have some side effects when the encodings of
          the categorical values is not contiguous ([1, 2, 10]) or there are different values in the synthetic
          dataframe.
        - For numerical columns, binning is determined by the numpy ``histogram_bin_edges`` function and takes into
          account values from BOTH dataframes.

        The final score is the average of the distances computed across columns. Lower is better.

        NOTE: The categorical columns MUST BE PREPROCESSED into numerical values otherwise the evaluation will fail.
        This function will NOT WORK WITH ONE-HOT ENCODINGS. This can be achieved by separately preprocessing the
        dataframes before calling compute or by setting ``do_preprocess`` to True.

        Args:
            categorical_columns: Column names corresponding to the categorical variables of any provided dataframe.
            numerical_columns: Column names corresponding to the numerical variables of any provided dataframe.
            do_preprocess: Whether or not to preprocess the dataframes with the default pipeline used by SynthEval.
                Defaults to False.
            include_numerical_columns: Whether to include any provided numerical columns in the Hellinger distance
                computation. Numerical column values are binned to create discrete distributions, which may or may not
                be something you want to do.
        """
        super().__init__(categorical_columns, numerical_columns, do_preprocess)

        self.include_numerical_columns = include_numerical_columns

    def compute(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict[str, float]:
        """
        Computes the DISCRETE Hellinger Distance between corresponding columns of real and synthetic dataframes. For a
        categorical column, the range of values for the discrete distributions is established by computing the unique
        values in the column for the REAL DATA. For numerical columns, a binning procedure based on numpy's
        ``histogram_bin_edges`` with binning strategy set to 'auto' is used.

        The final score is the average of the distances computed across columns. Lower is better.

        NOTE: The categorical columns MUST BE PREPROCESSED into numerical values otherwise the evaluation will fail.
        This function will NOT WORK WITH ONE-HOT ENCODINGS. This can be achieved by separately preprocessing the
        dataframes before calling compute or by setting ``do_preprocess`` to True.

        Args:
            real_data: Real data to which the synthetic data may be compared. In many cases this will be data used
                to TRAIN the model that generated the synthetic data, but not always.
            synthetic_data: Synthetically generated data whose quality is to be assessed.

        Returns:
            The mean of the individual Hellinger distances between each of the corresponding columns of the real and
            synthetic dataframes. This mean is keyed by 'mean_hellinger_distance' and is reported along with the
            "standard error" associated with that mean keyed under 'hellinger_standard_error'.
        """
        if self.do_preprocess:
            real_data, synthetic_data = self.preprocess(real_data, synthetic_data)

        hellinger_distances = []

        for category_column in self.categorical_columns:
            class_num = len(np.unique(real_data[category_column]))

            real_discrete_counts = np.histogram(real_data[category_column], bins=class_num)[0]
            synthetic_discrete_counts = np.histogram(synthetic_data[category_column], bins=class_num)[0]

            real_discrete_pdf = real_discrete_counts / sum(real_discrete_counts)
            synthetic_discrete_pdf = synthetic_discrete_counts / sum(synthetic_discrete_counts)

            distance = hellinger_distance(real_discrete_pdf, synthetic_discrete_pdf)
            hellinger_distances.append(distance)

        if self.include_numerical_columns:
            for numeric_column in self.numerical_columns:
                combined_data = np.concatenate((real_data[numeric_column], synthetic_data[numeric_column]))
                bin_edges = np.histogram_bin_edges(combined_data, bins="auto")

                real_discrete_counts = np.histogram(real_data[numeric_column], bins=bin_edges)[0]
                synthetic_discrete_counts = np.histogram(synthetic_data[numeric_column], bins=bin_edges)[0]

                real_discrete_pdf = real_discrete_counts / sum(real_discrete_counts)
                synthetic_discrete_pdf = synthetic_discrete_counts / sum(synthetic_discrete_counts)

                distance = hellinger_distance(real_discrete_pdf, synthetic_discrete_pdf)
                hellinger_distances.append(distance)

        mean_hellinger_distance = np.mean(hellinger_distances).item()
        hellinger_distance_standard_error = np.std(hellinger_distances, ddof=1).item() / np.sqrt(
            len(hellinger_distances)
        )

        return {
            "mean_hellinger_distance": mean_hellinger_distance,
            "hellinger_standard_error": hellinger_distance_standard_error,
        }
