import pickle
from logging import INFO
from pathlib import Path

import hydra
from omegaconf import DictConfig

from midst_toolkit.common.config import ClassifierConfig, ClusteringConfig, DiffusionConfig
from midst_toolkit.common.logger import TOOLKIT_LOGGER, log
from midst_toolkit.common.variables import DEVICE
from midst_toolkit.models.clavaddpm.clustering import clava_clustering
from midst_toolkit.models.clavaddpm.data_loaders import Table, load_tables
from midst_toolkit.models.clavaddpm.train import ModelArtifacts, clava_training


# Preventing some excessive logging
TOOLKIT_LOGGER.setLevel(INFO)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """
    Run the training pipeline.

    It will load the config and then data from the `config.base_data_dir` folder,
    train the model and save the results in the `config.results_dir` folder.

    Args:
        config: Training configuration as an OmegaConf DictConfig object.
    """
    log(INFO, f"Loading data from {config.base_data_dir}...")
    tables, relation_order, _ = load_tables(Path(config.base_data_dir))

    log(INFO, "Clustering data...")
    clustering_config = ClusteringConfig(**config.clustering_config)
    tables, _ = clava_clustering(tables, relation_order, Path(config.results_dir), clustering_config)

    log(INFO, "Training model...")
    diffusion_config = DiffusionConfig(**config.diffusion_config)
    classifier_config = ClassifierConfig(**config.classifier_config)

    tables, _ = clava_training(
        tables,
        relation_order,
        Path(config.results_dir),
        diffusion_config,
        classifier_config,
        device=DEVICE,
    )
    log(INFO, "Model trained successfully.")

    log(INFO, "Checking the clustering results...")
    clustering_results_file = Path(config.results_dir) / "cluster_ckpt.pkl"
    with open(clustering_results_file, "rb") as f:
        clustering_result = pickle.load(f)

    assert all(isinstance(table, Table) for table in clustering_result["tables"].values())
    assert isinstance(clustering_result["all_group_lengths_prob_dicts"], dict)

    for relation in relation_order:
        results_file = Path(config.results_dir) / "models" / f"{relation[0]}_{relation[1]}_ckpt.pkl"
        log(INFO, f"Checking the results from {results_file}...")

        with open(results_file, "rb") as f:
            result = pickle.load(f)

        # Asserting the results are the correct type
        assert isinstance(result, ModelArtifacts)

        log(INFO, f"Result size (in bytes): {results_file.stat().st_size}")


if __name__ == "__main__":
    main()
