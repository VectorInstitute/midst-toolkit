from logging import INFO
from pathlib import Path

import hydra
from ctgan import CTGAN  # type: ignore[import-untyped]
from omegaconf import DictConfig

from examples.gan.train import main as train_main
from examples.gan.utils import get_table_name
from midst_toolkit.common.logger import log


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """
    Run the synthesizing pipeline for a single-table CTGAN model.

    It will load the config and then data from the `config.base_data_dir` folder,
    load the trained model (or train one if it doesn't exist) and save the results
    in the `config.results_dir` folder.

    Args:
        config: Configuration as an OmegaConf DictConfig object.
    """
    results_file = Path(config.results_dir) / "trained_ctgan_model.pkl"

    if not results_file.exists():
        log(INFO, f"Trained model not found at {results_file}. Training a new model from scratch.")
        train_main(config)

    log(INFO, f"Loading model from {results_file}...")
    ctgan = CTGAN.load(results_file)

    log(INFO, f"Synthesizing data of size {config.synthesizing.sample_size}...")
    synthetic_data = ctgan.sample(config.synthesizing.sample_size)

    table_name = get_table_name(config.base_data_dir)
    synthetic_data_file = Path(config.results_dir) / f"{table_name}_synthetic.csv"
    log(INFO, f"Saving synthetic data to {synthetic_data_file}...")
    synthetic_data.to_csv(synthetic_data_file, index=False)

    log(INFO, "Done!")


if __name__ == "__main__":
    main()
