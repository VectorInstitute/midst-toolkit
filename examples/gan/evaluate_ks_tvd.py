import json
from logging import INFO
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from examples.gan.utils import get_table_name
from midst_toolkit.common.logger import log
from midst_toolkit.evaluation.quality.kolmogorov_smirnov_total_variation import KolmogorovSmirnovAndTotalVariation


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """
    Run the evaluation pipeline for the Kolmogorov-Smirnov and Total Variation Distance metrics.

    It will load the config and then data from the `config.base_data_dir` folder for the table
    name (from the `dataset_meta.json` file) and the real data under `{table_name}.csv`, and
    the synthetic data from the `config.results_dir` folder under `{table_name}_synthetic.csv`,
    and then compute the Kolmogorov-Smirnov and Total Variation Distance metrics.

    It will also need the meta_info.json file for the information about categorical and numerical
    columns.

    The results will be saved in the `config.results_dir` folder under `ks_tvd_evaluation.json`.

    Args:
        config: Configuration as an OmegaConf DictConfig object.
    """
    log(INFO, "Loading data...")

    table_name = get_table_name(config.base_data_dir)

    real_data = pd.read_csv(Path(config.base_data_dir) / f"{table_name}.csv")
    synthetic_data = pd.read_csv(Path(config.results_dir) / f"{table_name}_synthetic.csv")

    with open(Path(config.base_data_dir) / "meta_info.json", "r") as f:
        meta_info = json.load(f)

    numerical_columns = [real_data.columns[i] for i in meta_info["num_col_idx"]]
    categorical_columns = [real_data.columns[i] for i in meta_info["cat_col_idx"]]

    metric = KolmogorovSmirnovAndTotalVariation(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        do_preprocess=False,
    )
    score = metric.compute(real_data, synthetic_data)
    log(INFO, f"Kolmogorov-Smirnov and Total Variation Distance score: {score}")

    log(INFO, "Saving results...")
    with open(Path(config.results_dir) / "ks_tvd_evaluation.json", "w") as f:
        json.dump(score, f, indent=4)

    log(INFO, "Done!")


if __name__ == "__main__":
    main()
