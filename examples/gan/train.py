import json
from logging import INFO
from pathlib import Path

import hydra
import pandas as pd
from ctgan import CTGAN  # type: ignore[import-untyped]
from omegaconf import DictConfig

from examples.gan.utils import get_table_name
from midst_toolkit.common.logger import log


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """
    Run the training pipeline for a single-table CTGAN model.

    It will load the config and then data from the `config.base_data_dir` folder,
    train the model and save the results in the `config.results_dir` folder.

    Args:
        config: Configuration as an OmegaConf DictConfig object.
    """
    log(INFO, "Loading data...")

    table_name = get_table_name(config.base_data_dir)

    with open(Path(config.base_data_dir) / f"{table_name}_domain.json", "r") as f:
        domain_info = json.load(f)

    real_data = pd.read_csv(Path(config.base_data_dir) / f"{table_name}.csv")

    # Names of the columns that are discrete
    discrete_columns = [key for key, value in domain_info.items() if value["type"] == "discrete"]

    log(INFO, "Fitting CTGAN...")

    ctgan = CTGAN(epochs=10)
    ctgan.fit(real_data, discrete_columns)

    log(INFO, "Saving model...")
    results_file = Path(config.results_dir) / "trained_ctgan_model.pkl"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    ctgan.save(results_file)
    log(INFO, f"Model saved to {results_file}")

    log(INFO, "Sampling data...")

    # Create synthetic data
    synthetic_data = ctgan.sample(1000)

    log(INFO, "Done!")

    log(INFO, f"Synthetic data:\n{synthetic_data}")


if __name__ == "__main__":
    main()
