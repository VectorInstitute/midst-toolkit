""" This file is an uncompleted example script for running the ensemble attack on MIDST challenge provided resources and data. """
from logging import INFO
import hydra
from omegaconf import DictConfig
from pathlib import Path
from src.midst_toolkit.attacks.ensemble.process_split_data import process_split_data
from src.midst_toolkit.common.logger import log
from examples.ensemble_attack_example.real_data_collection import collect_population_data_ensemble


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):

    if cfg.pipeline.run_data_processing:
        log(INFO, "Running data processing pipeline...")
        # Collect the real data from the MIDST challenge resources.
        population_data = collect_population_data_ensemble(
            midst_data_input_dir=Path(cfg.data_paths.midst_data_path),
            data_processing_config=cfg.data_processing_config,
            save_dir=Path(cfg.data_paths.population_path),
        )
        # The following function saves the required dataframe splits in the specified processed_attack_data_path path.
        process_split_data(
            all_population_data=population_data,
            processed_attack_data_path=Path(cfg.data_paths.processed_attack_data_path),
            # TODO: column_to_stratify value is not documented in the original codebase.
            column_to_stratify=cfg.data_processing_config.column_to_stratify,
            num_total_samples=cfg.data_processing_config.population_sample_size,
            random_seed=cfg.random_seed,
        )
        log(INFO, "Data processing pipeline finished.")


if __name__ == "__main__":
    main()
