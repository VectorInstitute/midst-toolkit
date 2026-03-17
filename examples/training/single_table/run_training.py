import pickle
from logging import INFO
from pathlib import Path

import hydra
from omegaconf import DictConfig

from midst_toolkit.common.config import ClavaDDPMDiffusionConfig
from midst_toolkit.common.logger import TOOLKIT_LOGGER, log
from midst_toolkit.common.variables import DEVICE
from midst_toolkit.models.clavaddpm.data_loaders import load_tables
from midst_toolkit.models.clavaddpm.train import ClavaDDPMModelArtifacts, clava_training


# Preventing some excessive logging
TOOLKIT_LOGGER.setLevel(INFO)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """
    Run the training pipeline for a single-table diffusion model.

    It will load the config and then data from the `config.base_data_dir` folder,
    train the model and save the results in the `config.results_dir` folder.

    Args:
        config: Training configuration as an OmegaConf DictConfig object.
    """
    log(INFO, f"Loading data from {config.base_data_dir}...")
    tables, relation_order, _ = load_tables(Path(config.base_data_dir))

    log(INFO, "Training model...")
    diffusion_config = ClavaDDPMDiffusionConfig(**config.diffusion_config)

    tables, _ = clava_training(
        tables,
        relation_order,
        Path(config.results_dir),
        diffusion_config,
        device=DEVICE,
    )
    log(INFO, "Model trained successfully.")

    results_file = Path(config.results_dir) / "models" / "None_trans_ckpt.pkl"
    log(INFO, f"Checking the results from {results_file}...")

    with open(results_file, "rb") as f:
        result = pickle.load(f)

    # Asserting the results are the correct type
    assert isinstance(result, ClavaDDPMModelArtifacts)

    log(INFO, f"Result size (in bytes): {results_file.stat().st_size}")


if __name__ == "__main__":
    main()
