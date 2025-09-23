from enum import Enum

import pandas as pd
from syntheval.metrics.privacy.metric_epsilon_identifiability import EpsilonIdentifiability

from midst_toolkit.evaluation.metrics_base import SynthEvalMetric


class EpsilonIdentifiabilityNorm(Enum):
    """These are the valid norms for SynthEval measures."""

    L2 = "euclid"
    GOWER = "gower"


class EpsilonIdentifiabilityRisk(SynthEvalMetric):
    def __init__(
        self,
        categorical_columns: list[str],
        numerical_columns: list[str],
        do_preprocess: bool = False,
        norm: EpsilonIdentifiabilityNorm = EpsilonIdentifiabilityNorm.GOWER,
    ):
        """
        Class to compute the Epsilon Identifiability Risk. This computes the ratio of real samples that have a
        synthetic data point closer than any other real data point in the set of samples. As such, a value closer to 0
        is better.

        If a holdout set is provided to the compute function, the same ratio is computed for holdout data points
        compared with synthetic ones. The difference between the ratio for the real samples compared with the
        holdout samples is then calculated. Ideally, these should be roughly the same (i.e. difference near zero) or
        negative. In this scenario, it is typical that the real data was USED TO TRAIN a model that generated the
        synthetic data and the holdout set represents real data that was NOT.

        NOTE: Dimensions are not uniformly weighted. They are weighted by their inverse column entropy to provide
        greater attention to rare data points. This is formally defined in

        Yoon, J., Drumright, L.N., Schaar, M.: Anonymization through data synthesis using generative adversarial
        networks (ADS-GAN). IEEE J. Biomed. Health Informatics 24(8), 2378–2388 (2020)
        https://doi.org/10.1109/JBHI.2020.2980262

        NOTE: The dataframes provided need to be pre-processed into numerical values for each column in some way. That
        is, for example, the categorical variables may be one-hot encoded and the numerical values normalized in
        some way. This can be done via the ``preprocess`` function in ``distance_preprocess.py`` beforehand or it can
        be done within compute if ``do_preprocess`` is True using the SynthEval pipeline.



        Args:
            categorical_columns: Column names corresponding to the categorical variables of any provided dataframe.
            numerical_columns: Column names corresponding to the numerical variables of any provided dataframe.
            do_preprocess: Whether or not to preprocess the dataframes with the default pipeline used by SynthEval.
                Defaults to False.
            norm: The kind of norm to use when measuring distances between points. Only l2 and Gower norms are
                currently supported. SynthEval defaults to Gower, so we do here as well. Note that if norm is
                EpsilonIdentifiabilityNorm.L2, then distances only consider the columns specified by
                ``numerical_columns``. Defaults to EpsilonIdentifiabilityNorm.GOWER.
        """
        super().__init__(categorical_columns, numerical_columns, do_preprocess)
        self.norm = norm
        self.all_columns = categorical_columns + numerical_columns

    def compute(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        holdout_data: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """
        Computes the Epsilon Identifiability Risk. This is the ratio of samples from ``real_data`` that have a point
        from ``synthetic_data` that is closer than any other real data point in ``real_data``. As such, a value
        closer to 0 is better.

        If ``holdout_data`` is provided, the same ratio is computed for points in ``holdout_data`` compared with those
        in ``synthetic_data``. The difference between the ratio for ``real_data`` compared with ``holdout_data`` is
        then calculated. Ideally, these should be roughly the same (i.e. difference near zero) or negative. In this
        scenario, it is typical that the real data was USED TO TRAIN a model that generated the synthetic data and the
        holdout set represents real data that was NOT.

        NOTE: Dimensions are not uniformly weighted. They are weighted by their inverse column entropy to provide
        greater attention to rare data points. This is formally defined in

        Yoon, J., Drumright, L.N., Schaar, M.: Anonymization through data synthesis using generative adversarial
        networks (ADS-GAN). IEEE J. Biomed. Health Informatics 24(8), 2378–2388 (2020)
        https://doi.org/10.1109/JBHI.2020.2980262

        NOTE: The dataframes provided need to be pre-processed into numerical values for each column in some way. That
        is, for example, the categorical variables may be one-hot encoded and the numerical values normalized in
        some way. This can be done via the ``preprocess`` function in ``distance_preprocess.py`` beforehand or it can
        be done within compute if ``do_preprocess`` is True using the SynthEval pipeline.

        Args:
            real_data: Real data to which the synthetic data may be compared. In many cases this will be data used
                to TRAIN the model that generated the synthetic data, but not always.
            synthetic_data: Synthetically generated data whose quality is to be assessed.
            holdout_data: Real data to which the synthetic data may also be compared. In many cases this will be data
                was NOT used in training the generating model. If none, then 'privacy_loss' is not computed.

        Returns:
            A dictionary of Epsilon Identifiability Risk results. Regardless of input, the estimated epsilon
            identifiability risk for ``real_data`` is reported, keyed by 'epsilon_identifiability_risk'. If
            ``holdout_data`` is provided. The difference of the risk for ``real_data`` and ``holdout_data`` is
            reported, keyed by 'privacy_loss'.
        """
        if self.do_preprocess:
            if holdout_data is None:
                real_data, synthetic_data = self.preprocess(real_data, synthetic_data)
            else:
                real_data, synthetic_data, holdout_data = self.preprocess(real_data, synthetic_data, holdout_data)

        # When using the l2 distance, SynthEval aims to filter to only the numerical columns. However, there is a bug
        # when providing a holdout set, where that set does not get filtered. So we'll do it here.
        if self.norm == EpsilonIdentifiabilityNorm.L2:
            filtered_real_data = real_data[self.numerical_columns]
            filtered_synthetic_data = synthetic_data[self.numerical_columns]
            filtered_holdout_data = holdout_data[self.numerical_columns] if holdout_data is not None else None
        else:
            # NOTE: The SynthEval class ignores column specifications by default. However, for other classes
            # (correlation_matrix_difference for example), specifying less than all of the columns restricts the score
            # computation to just those columns. To make this consistent we do that here, before passing to the
            # SynthEval class.
            filtered_real_data = real_data[self.all_columns]
            filtered_synthetic_data = synthetic_data[self.all_columns]
            filtered_holdout_data = holdout_data[self.all_columns] if holdout_data is not None else None

        self.syntheval_metric = EpsilonIdentifiability(
            real_data=filtered_real_data,
            synt_data=filtered_synthetic_data,
            hout_data=filtered_holdout_data,
            cat_cols=self.categorical_columns,
            num_cols=self.numerical_columns,
            do_preprocessing=False,
            verbose=False,
            nn_dist=self.norm.value,
        )
        result = self.syntheval_metric.evaluate()
        result["epsilon_identifiability_risk"] = result.pop("eps_risk")
        if holdout_data is not None:
            result["privacy_loss"] = result.pop("priv_loss")
        return result
