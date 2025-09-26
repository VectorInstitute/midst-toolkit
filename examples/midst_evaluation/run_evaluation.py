from logging import INFO
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig

from examples.midst_evaluation.preprocessing import (
    get_numerical_and_categorical_column_names,
    preprocess_data_for_alpha_precision_eval,
    syntheval_preprocess,
)
from midst_toolkit.common.logger import log
from midst_toolkit.data_processing.midst_data_processing import load_midst_data_with_test
from midst_toolkit.evaluation.metrics_base import MetricBase
from midst_toolkit.evaluation.privacy import (
    DistanceToClosestRecordScore,
    EpsilonIdentifiabilityRisk,
    HittingRate,
    MedianDistanceToClosestRecordScore,
    NearestNeighborDistanceRatio,
)
from midst_toolkit.evaluation.privacy.distance_preprocess import preprocess
from midst_toolkit.evaluation.privacy.distance_utils import NormType
from midst_toolkit.evaluation.privacy.epsilon_identifiability_risk import EpsilonIdentifiabilityNorm
from midst_toolkit.evaluation.quality import (
    AlphaPrecision,
    CorrelationMatrixDifference,
    DimensionwiseMeanDifference,
    KolmogorovSmirnovAndTotalVariation,
    MeanConfidenceIntervalOverlap,
    MeanF1ScoreDifference,
    MeanHellingerDistance,
    MeanPropensityMeanSquaredError,
    MutualInformationDifference,
)
from midst_toolkit.evaluation.quality.confidence_interval_overlap import ConfidenceLevel


def log_metrics(header: str, results: dict[str, float]) -> None:
    """
    Helper function to log metrics associated with the results dictionary in a structured fashion. The header
    is used to separate out different families of metrics in the output.

    Args:
        header: String to describe the set of metrics that will be logged.
        results: Dictionary of metric names (keys) and metric values (values) to be logged.
    """
    log(INFO, f"\n{header}\n------------------------------------------------------------------------------------\n")
    for metric_name, metric_value in results.items():
        log(INFO, f"Metric: {metric_name}\tMetric: {metric_value}")
    log(INFO, "------------------------------------------------------------------------------------\n")


def write_metrics(metric_report_path: Path, header: str, results: dict[str, float]) -> None:
    """
    Helper function to write metrics associated with the results dictionary in a structured fashion to a file. The
    header is used to separate out different families of metrics in the output.

    Args:
        metric_report_path: Path to with the metrics will be written
        header: String to describe the set of metrics that will be written.
        results: Dictionary of metric names (keys) and metric values (values) to be written.
    """
    with open(metric_report_path, "a") as f:
        f.write(f"\n{header}\n------------------------------------------------------------------------------------\n")
        for metric_name, metric_value in results.items():
            f.write(f"Metric: {metric_name:40}Metric: {metric_value}\n")
        f.write("------------------------------------------------------------------------------------\n")


def report_metrics(cfg: DictConfig, header: str, results: dict[str, float]) -> None:
    """
    A helper function facilitate both logging and, optionally depending on the configuration settings, write the
    metrics results to a file.

    Args:
        cfg: Configuration determining if and where to write metrics to a file.
        header: String to describe the set of metrics that will be logged/written.
        results: Dictionary of metric names (keys) and metric values (values) to be logged/written.
    """
    log_metrics(header, results)
    if cfg.write_report:
        write_metrics(Path(cfg.metric_report_path), header, results)


def remove_label_column_from_other_columns(
    label_column: str, numerical_columns: list[str], categorical_columns: list[str]
) -> tuple[list[str], list[str]]:
    """
    Given a column name for a target label (task label), this ensures that the label is removed from either the
    numerical or categorical columns list. During preprocessing, it is advantageous to also process the label
    column into, for instance, and ordinal value. However, when performing F1 measurements, we no longer want it
    to be part of the dataframe during training.

    Args:
        label_column: Column name associated with task labels of interest.
        numerical_columns: Set of column names associated with numerical values.
        categorical_columns: Set of column names associated with categorical values.

    Raises:
        ValueError: Will throw an error if the label column is present in both column names lists, which is bad...

    Returns:
        Filtered copies of the numerical and categorical column names without the specified label column included.
    """
    if label_column in numerical_columns and label_column in categorical_columns:
        raise ValueError("Label column appears in both types of columns...")
    return [item for item in numerical_columns if label_column != item], [
        item for item in categorical_columns if label_column != item
    ]


def should_syntheval_preprocess(cfg: DictConfig, for_privacy: bool) -> bool:
    """
    Determines whether, based on the configuration, syntheval preprocessing needs to be performed.

    Args:
        cfg: Configuration with the settings for the entire evaluation pipeline.
        for_privacy: Whether this is for the privacy evaluations (True) or for the quality evaluations (False).

    Returns:
        True if preprocessing with the SynthEval pipeline should be performed, False otherwise
    """
    if for_privacy:
        return any([cfg.hitting_rate.run, cfg.eir.run])
    return any(
        [
            cfg.ks_tv.run,
            cfg.ci_overlap.run,
            cfg.ks_tv.run,
            cfg.correlation_diff.run,
            cfg.mean_diff.run,
            cfg.f1_score_diff.run,
            cfg.hellinger.run,
            cfg.propensity_mse.run,
            cfg.mutual_information.run,
        ]
    )


def run_quality_evaluations(
    # ruff: noqa: PLR0915
    cfg: DictConfig,
    real_data_train: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    real_data_test: pd.DataFrame,
    meta_info: dict[str, Any],
) -> None:
    """
    Run quality evaluation metrics.

    Args:
        cfg: Configuration information for the evaluation.
        real_data_train: Dataframe containing real data to which the synthetic data will be compared (typically used to
            train the model that generation the synthetic data).
        synthetic_data: Dataframe containing the synthetic data whose quality is to be measured.
        real_data_test: Dataframe containing real data to which the synthetic data will be compared (typically
            explicitly NOT USED to train the model that generation the synthetic data).
        meta_info: Dictionary containing information about the dataframes, including which columns correspond to
            numerical and categorical values or whether there is a column corresponding to a label.
    """
    metric: MetricBase
    if cfg.alpha_precision.run:
        log(INFO, "Preprocessing Data for Alpha Precision Evaluation")
        # Categorical values are one-hot encoded, numerical values are left alone.
        alpha_precision_real_data, alpha_precision_synthetic_data = preprocess_data_for_alpha_precision_eval(
            real_data=real_data_train, synthetic_data=synthetic_data, meta_info=meta_info
        )
        log(INFO, "Running Alpha-Precision Evaluation")
        metric = AlphaPrecision(naive_only=cfg.alpha_precision.naive_only)
        results = metric.compute(alpha_precision_real_data, alpha_precision_synthetic_data)
        report_metrics(cfg, "ALPHA PRECISION", results)

    # Shared preprocessing for syntheval based metrics if they are to be run
    if should_syntheval_preprocess(cfg, for_privacy=False):
        log(INFO, "Preprocessing Data with SynthEval pipeline")
        numerical_columns, categorical_columns = get_numerical_and_categorical_column_names(real_data_train, meta_info)

        # Categorical values are ordinal encoded, numerical values are min-max scaled
        syntheval_real_data_train, syntheval_synthetic_data, syntheval_real_data_test = syntheval_preprocess(
            numerical_columns, categorical_columns, real_data_train, synthetic_data, real_data_test
        )

    if cfg.ks_tv.run:
        log(INFO, "Running Kolmogorov-Smirnov and Total Variation Evaluation")
        metric = KolmogorovSmirnovAndTotalVariation(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            significance_level=cfg.ks_tv.significance_level,
            permutations=cfg.ks_tv.permutations,
            # Already preprocessing above
            do_preprocess=False,
        )
        results = metric.compute(syntheval_real_data_train, syntheval_synthetic_data)
        report_metrics(cfg, "KOLMOGOROV SMIRNOV AND TOTAL VARIATION", results)

    if cfg.ci_overlap.run:
        log(INFO, "Running Confidence Interval Overlap Evaluation")
        metric = MeanConfidenceIntervalOverlap(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            confidence_level=ConfidenceLevel(cfg.ci_overlap.confidence_level),
            # Already preprocessing above
            do_preprocess=False,
        )
        results = metric.compute(syntheval_real_data_train, syntheval_synthetic_data)
        report_metrics(cfg, "CONFIDENCE INTERVAL OVERLAP", results)

    if cfg.correlation_diff.run:
        log(INFO, "Running Mean Correlation Matrix Difference Evaluation")
        metric = CorrelationMatrixDifference(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            compute_mixed_correlations=cfg.correlation_diff.compute_mixed_correlations,
            # Already preprocessing above
            do_preprocess=False,
        )
        results = metric.compute(syntheval_real_data_train, syntheval_synthetic_data)
        report_metrics(cfg, "CORRELATION MATRIX DIFFERENCE", results)

    if cfg.mean_diff.run:
        log(INFO, "Running Dimensionwise Mean Difference Evaluation")
        metric = DimensionwiseMeanDifference(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            # Already preprocessing above
            do_preprocess=False,
        )
        results = metric.compute(syntheval_real_data_train, syntheval_synthetic_data)
        report_metrics(cfg, "DIMENSIONWISE MEAN DIFFERENCE", results)

    if cfg.f1_score_diff.run:
        # Explicitly removing the target/label column from other column names
        label_column = cfg.f1_score_diff.label_column
        filtered_numerical_columns, filtered_categorical_columns = remove_label_column_from_other_columns(
            label_column, numerical_columns, categorical_columns
        )
        log(INFO, "Running F1 Score Difference Evaluation")
        metric = MeanF1ScoreDifference(
            categorical_columns=filtered_categorical_columns,
            numerical_columns=filtered_numerical_columns,
            label_column=label_column,
            folds=cfg.f1_score_diff.folds,
            f1_type=cfg.f1_score_diff.f1_type,
            # Already preprocessing above
            do_preprocess=False,
        )
        results = metric.compute(syntheval_real_data_train, syntheval_synthetic_data, syntheval_real_data_test)
        report_metrics(cfg, "F1 SCORE DIFFERENCE", results)

    if cfg.hellinger.run:
        log(INFO, "Running Hellinger Distance Difference Evaluation")
        metric = MeanHellingerDistance(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            include_numerical_columns=cfg.hellinger.include_numerical_columns,
            # Already preprocessing above
            do_preprocess=False,
        )
        results = metric.compute(syntheval_real_data_train, syntheval_synthetic_data)
        report_metrics(cfg, "HELLINGER DISTANCE DIFFERENCE", results)

    if cfg.propensity_mse.run:
        log(INFO, "Running Propensity Mean Squared Error Evaluation")
        metric = MeanPropensityMeanSquaredError(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            folds=cfg.propensity_mse.folds,
            max_iterations=cfg.propensity_mse.max_iterations,
            solver=cfg.propensity_mse.solver,
            # Already preprocessing above
            do_preprocess=False,
        )
        results = metric.compute(syntheval_real_data_train, syntheval_synthetic_data)
        report_metrics(cfg, "PROPENSITY MEAN SQUARED ERROR", results)

    if cfg.mutual_information.run:
        log(INFO, "Running Mutual Information Difference Evaluation")
        metric = MutualInformationDifference(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            include_numerical_columns=cfg.mutual_information.include_numerical_columns,
            # Already preprocessing above
            do_preprocess=False,
        )
        results = metric.compute(syntheval_real_data_train, syntheval_synthetic_data)
        report_metrics(cfg, "MUTUAL INFORMATION DIFFERENCE", results)


def run_privacy_evaluations(
    cfg: DictConfig,
    real_data_train: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    real_data_test: pd.DataFrame,
    meta_info: dict[str, Any],
) -> None:
    """
    Run Privacy evaluation metrics.

    Args:
        cfg: Configuration information for the evaluation.
        real_data_train: Dataframe containing real data to which the synthetic data will be compared (typically used to
            train the model that generation the synthetic data).
        synthetic_data: Dataframe containing the synthetic data whose quality is to be measured.
        real_data_test: Dataframe containing real data to which the synthetic data will be compared (typically
            explicitly NOT USED to train the model that generation the synthetic data).
        meta_info: Dictionary containing information about the dataframes, including which columns correspond to
            numerical and categorical values or whether there is a column corresponding to a label.
    """
    metric: MetricBase
    # Shared preprocessing for syntheval based metrics if they are to be run
    if should_syntheval_preprocess(cfg, for_privacy=True):
        log(INFO, "Preprocessing Data with SynthEval pipeline")
        numerical_columns, categorical_columns = get_numerical_and_categorical_column_names(real_data_train, meta_info)

        # Categorical values are ordinal encoded, numerical values are min-max scaled
        syntheval_real_data_train, syntheval_synthetic_data = syntheval_preprocess(
            numerical_columns, categorical_columns, real_data_train, synthetic_data
        )

    if cfg.hitting_rate.run:
        log(INFO, "Running Hitting Rate Evaluation")

        metric = HittingRate(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            hitting_threshold=cfg.hitting_rate.hitting_threshold,
            # Already preprocessing above
            do_preprocess=False,
        )
        results = metric.compute(syntheval_real_data_train, syntheval_synthetic_data)
        report_metrics(cfg, "HITTING RATE", results)

    if cfg.eir.run:
        log(INFO, "Running Epsilon Identifiability Rate Evaluation")
        # Categorical values are ordinal encoded, numerical values are min-max scaled
        metric = EpsilonIdentifiabilityRisk(
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            norm=EpsilonIdentifiabilityNorm(cfg.eir.norm),
            # Already preprocessing above
            do_preprocess=False,
        )
        results = metric.compute(syntheval_real_data_train, syntheval_synthetic_data)
        report_metrics(cfg, "EPSILON IDENTIFIABILITY RISK", results)

    # Shared preprocessing for distance based metrics, if they are to be run.
    if any([cfg.dcr.run, cfg.median_dcr.run, cfg.nndr.run]):
        log(INFO, "Preprocessing Data for Distance Evaluation")
        # Categorical values are one-hot encoded, numerical values are scaled by their range, but not into [0,1]
        distance_real_data, distance_synthetic_data, distance_holdout_data = preprocess(
            meta_info=meta_info,
            synthetic_data=synthetic_data,
            real_data_train=real_data_train,
            real_data_test=real_data_test,
        )

    if cfg.dcr.run:
        log(INFO, "Running DCR Evaluation")
        metric = DistanceToClosestRecordScore(NormType(cfg.dcr.norm), cfg.dcr.batch_size, do_preprocess=False)
        results = metric.compute(distance_real_data, distance_synthetic_data, distance_holdout_data)
        report_metrics(cfg, "DISTANCE TO CLOSEST RECORD", results)

    if cfg.median_dcr.run:
        log(INFO, "Running Median DCR Evaluation")
        metric = MedianDistanceToClosestRecordScore(
            NormType(cfg.median_dcr.norm), cfg.median_dcr.batch_size, do_preprocess=False
        )
        results = metric.compute(distance_real_data, distance_synthetic_data)
        report_metrics(cfg, "MEDIAN DISTANCE TO CLOSEST RECORD", results)

    if cfg.nndr.run:
        log(INFO, "Running NNDR Evaluation")
        metric = NearestNeighborDistanceRatio(NormType(cfg.nndr.norm), cfg.nndr.batch_size, do_preprocess=False)
        results = metric.compute(distance_real_data, distance_synthetic_data, distance_holdout_data)
        report_metrics(cfg, "NEAREST NEIGHBOR DISTANCE RATIO", results)


@hydra.main(config_path=".", config_name="config_adult", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point for the evaluation script."""
    log(INFO, "Loading Data for Evaluations")
    real_data_train, synthetic_data, real_data_test, meta_info = load_midst_data_with_test(
        Path(cfg.data_paths.real_train_data_path),
        Path(cfg.data_paths.synthetic_data_path),
        cfg.data_paths.meta_data_path,
        Path(cfg.data_paths.real_test_data_path),
    )

    run_quality_evaluations(cfg, real_data_train, synthetic_data, real_data_test, meta_info)
    run_privacy_evaluations(cfg, real_data_train, synthetic_data, real_data_test, meta_info)


if __name__ == "__main__":
    main()
