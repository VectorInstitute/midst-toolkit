import importlib
from pathlib import Path
from logging import INFO

from omegaconf import DictConfig
import hydra

from midst_toolkit.common.logger import log
from midst_toolkit.common.random import set_all_random_seeds
from midst_toolkit.attacks.ensemble.process_split_data import process_split_data
from examples.ensemble_attack.real_data_collection import collect_population_data_ensemble



@hydra.main(config_path="../", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """
    Run the Ensemble Attack pipeline with the CTGAN model.

    As the first step, data processing is done.
    Second step is shadow model training used for RMIA attack.
    Third step is metaclassifier training and evaluation.

    Args:
        config: Attack configuration as an OmegaConf DictConfig object.
    """

    import ipdb;ipdb.set_trace()

    if config.ensemble_attack.random_seed is not None:
        set_all_random_seeds(seed=config.ensemble_attack.random_seed)
        log(INFO, f"Training phase random seed set to {config.ensemble_attack.random_seed}.")

    # Note: Importing the following two modules causes a segmentation fault error if imported together in this file.
    # A quick solution is to load modules dynamically if any of the pipelines is called.
    # TODO: Investigate the source of error.
    shadow_pipeline = importlib.import_module("examples.ensemble_attack.run_shadow_model_training")
    shadow_data_paths = shadow_pipeline.run_shadow_model_training(config.ensemble_attack)
    shadow_data_paths = [Path(path) for path in shadow_data_paths]

    target_model_synthetic_path = shadow_pipeline.run_target_model_training(config)

    if config.pipeline.run_metaclassifier_training:
        if not config.pipeline.run_shadow_model_training:
            # If shadow model training is skipped, we need to provide the previous shadow model and target model paths.
            shadow_data_paths = [Path(path) for path in config.shadow_training.final_shadow_models_path]
            target_model_synthetic_path = Path(config.shadow_training.target_synthetic_data_path)

        assert len(shadow_data_paths) == 3, "The attack_data_paths list must contain exactly three elements."
        assert target_model_synthetic_path is not None, (
            "The target_data_path must be provided for metaclassifier training."
        )

        meta_pipeline = importlib.import_module("examples.ensemble_attack.run_metaclassifier_training")
        meta_pipeline.run_metaclassifier_training(config, shadow_data_paths, target_model_synthetic_path)


if __name__ == "__main__":
    main()
