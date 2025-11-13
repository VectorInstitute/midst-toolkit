import pickle
from logging import INFO
from pathlib import Path

import hydra
from omegaconf import DictConfig

from examples.training.single_table import run_training
from midst_toolkit.common.config import GeneralConfig, MatchingConfig, SamplingConfig
from midst_toolkit.common.logger import TOOLKIT_LOGGER, log
from midst_toolkit.models.clavaddpm.data_loaders import load_tables
from midst_toolkit.models.clavaddpm.synthesizer import clava_synthesizing


# Preventing some excessive logging
TOOLKIT_LOGGER.setLevel(INFO)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """
    Run the synthesizing pipeline for a single-table diffusion model.

    It will load the config and then data from the `config.base_data_dir` folder,
    train the model, synthesize the data and save the results in the
    `config.results_dir` folder.

    It will first look for a pre-trained model in the `config.results_dir` folder.
    If it doesn't find one, it will train a new model from scratch.

    Args:
        config: Training configuration as an OmegaConf DictConfig object.
    """
    log(INFO, f"Checking for a pre-trained model in {config.results_dir}...")

    tables, relation_order, _ = load_tables(Path(config.base_data_dir))

    model_file_paths = {}
    for relation in relation_order:
        model_file_path = Path(config.results_dir) / "models" / f"{relation[0]}_{relation[1]}_ckpt.pkl"
        model_file_paths[relation] = model_file_path

    if all(model_file.exists() for model_file in model_file_paths.values()):
        log(INFO, f"Found a pre-trained models in {config.results_dir}. Skipping training.")
    else:
        log(INFO, "No pre-trained models found, training a new model from scratch...")
        run_training.main(config)

    log(INFO, "Synthesizing data...")

    models = {}
    for relation in relation_order:
        with open(model_file_paths[relation], "rb") as f:
            models[relation] = pickle.load(f)

    clava_synthesizing(
        tables,
        relation_order,
        Path(config.results_dir),
        models,
        GeneralConfig(**config.general_config),
        SamplingConfig(**config.sampling_config),
        MatchingConfig(**config.matching_config),
    )

    log(INFO, "Data synthesized successfully.")


if __name__ == "__main__":
    main()
