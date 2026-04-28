import json
from logging import INFO
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from midst_toolkit.common.logger import log
from midst_toolkit.evaluation.quality.correlation_matrix_difference import CorrelationMatrixDifference
from midst_toolkit.evaluation.quality.kolmogorov_smirnov_total_variation import KolmogorovSmirnovAndTotalVariation
from midst_toolkit.evaluation.quality.mutual_information_difference import MutualInformationDifference


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """
    Run the evaluation pipeline for the Kolmogorov-Smirnov and Total Variation Distance metrics.

    It will load the data from the `config.data_dir` folder for the table
    name (from `config.table_name`) and the real data under `{table_name}.csv`, and
    the synthetic data from the `results_dir/{table_name}/synthetic_data` folder under
    `{table_name}_synthetic.csv`, and then compute the Kolmogorov-Smirnov and
    Total Variation Distance metrics.

    It will also need the config.data_dir/{table_name}.json file for the information about
    categorical and numerical columns.

    The results will be saved in the `config.results_dir` folder under `ks_tvd_evaluation.json`.

    Args:
        config: Configuration as an OmegaConf DictConfig object.
    """
    log(INFO, "Loading data...")

    real_data = pd.read_csv(Path(config.data_dir) / f"{config.table_name}.csv")
    synthetic_data = pd.read_csv(
        Path(config.results_dir) / config.table_name / "synthetic_data" / f"{config.table_name}_synthetic.csv"
    )

    with open(Path(config.data_dir) / f"{config.table_name}.json", "r") as f:
        data_info = json.load(f)

    numerical_columns = [real_data.columns[i] for i in data_info["num_col_idx"]]
    categorical_columns = [real_data.columns[i] for i in data_info["cat_col_idx"]]

    results = {}

    # KS and TVD
    ks_tvd_metric = KolmogorovSmirnovAndTotalVariation(categorical_columns, numerical_columns, do_preprocess=True)
    ks_tvd_score = ks_tvd_metric.compute(real_data, synthetic_data)

    log(INFO, f"Kolmogorov-Smirnov and Total Variation Distance score: {ks_tvd_score}")
    results["ks_tvd"] = ks_tvd_score

    # Correlation Matrix Difference
    cmd_metric = CorrelationMatrixDifference(categorical_columns, numerical_columns, do_preprocess=True)
    cmd_result = cmd_metric.compute(real_data, synthetic_data)

    log(INFO, f"Correlation Matrix Difference score: {cmd_result}")
    results["correlation_matrix_difference"] = cmd_result

    # Mutual Information Difference
    mid_metric = MutualInformationDifference(categorical_columns, numerical_columns, do_preprocess=True)
    mid_result = mid_metric.compute(real_data, synthetic_data)
    mid_result["score"] = mid_result["mutual_inf_diff"] / mid_result["mi_mat_dims"]

    log(INFO, f"Mutual Information Difference score: {mid_result}")
    results["mutual_information_difference"] = mid_result

    log(INFO, "Saving results...")
    with open(Path(config.results_dir) / "evaluation.json", "w") as f:
        json.dump(results, f, indent=4)

    log(INFO, "Done!")


if __name__ == "__main__":
    main()
