import pickle
from logging import INFO
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from examples.training.multi_table import run_training
from midst_toolkit.common.config import GeneralConfig, MatchingConfig, SamplingConfig
from midst_toolkit.common.logger import TOOLKIT_LOGGER, log
from midst_toolkit.models.clavaddpm.data_loaders import load_tables
from midst_toolkit.models.clavaddpm.enumerations import Relation
from midst_toolkit.models.clavaddpm.synthesizer import clava_synthesizing


# Preventing some excessive logging
TOOLKIT_LOGGER.setLevel(INFO)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """
    Run the synthesizing pipeline for a multi-table diffusion model.

    It will load the config and then data from the `config.base_data_dir` folder,
    train the model, synthesize the data and save the results in the
    `config.results_dir` folder.

    It will first look for a pre-trained model in the `config.results_dir` folder.
    If it doesn't find one, it will train a new model from scratch.

    Args:
        config: Training and synthesizing configuration as an OmegaConf DictConfig object.
    """
    log(INFO, f"Checking for a pre-trained model in {config.results_dir}...")

    _, relation_order, _ = load_tables(Path(config.base_data_dir))

    model_file_paths: dict[Relation, dict[str, Any]] = {}
    for relation in relation_order:
        model_file_path = Path(config.results_dir) / "models" / f"{relation[0]}_{relation[1]}_ckpt.pkl"
        model_file_paths[relation] = {
            "file_path": model_file_path,
            "exists": model_file_path.exists(),
        }

    clustering_results_file = Path(config.results_dir) / "cluster_ckpt.pkl"

    if all(result["exists"] for result in model_file_paths.values()) and clustering_results_file.exists():
        log(INFO, f"Found previous results in {config.results_dir}. Skipping training.")
    else:
        log(INFO, "Not all previous results found. Training a new model from scratch.")
        log(INFO, f"Summary of results: {model_file_paths}")
        log(INFO, f"Clustering results file: {clustering_results_file} exists? {clustering_results_file.exists()}")
        run_training.main(config)

    log(INFO, "Loading models...")

    models = {}
    for relation in relation_order:
        with open(model_file_paths[relation]["file_path"], "rb") as f:
            models[relation] = pickle.load(f)

    with open(clustering_results_file, "rb") as f:
        clustering_result = pickle.load(f)

    tables = clustering_result["tables"]
    all_group_lengths_prob_dicts = clustering_result["all_group_lengths_prob_dicts"]

    log(INFO, "Synthesizing data...")

    clava_synthesizing(
        tables,
        relation_order,
        Path(config.results_dir),
        models,
        GeneralConfig(**config.general_config),
        SamplingConfig(**config.sampling_config),
        MatchingConfig(**config.matching_config),
        all_group_lengths_prob_dicts,
    )

    log(INFO, "Data synthesized successfully.")


if __name__ == "__main__":
    main()
