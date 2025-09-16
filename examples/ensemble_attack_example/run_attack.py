"""
This file is an uncompleted example script for running the Ensemble Attack on MIDST challenge
provided resources and data.
"""

from logging import INFO
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from examples.ensemble_attack_example.real_data_collection import collect_population_data_ensemble
from midst_toolkit.attacks.ensemble.blending import BlendingPlusPlus
from midst_toolkit.attacks.ensemble.data_utils import load_dataframe
from midst_toolkit.attacks.ensemble.process_split_data import process_split_data
from midst_toolkit.common.logger import log


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Run the Ensemble Attack example pipeline.
    As the first step, data processing is done.

    Args:
        cfg: Attack OmegaConf DictConfig object.
    """
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

    elif cfg.pipeline.run_metaclassifier_training:
        log(INFO, "Running metaclassifier training...")
        # Load the processed data splits.
        df_meta_train = load_dataframe(
            Path(cfg.data_paths.processed_attack_data_path),
            "og_train_meta.csv",
        )
        y_meta_train = np.load(
            Path(cfg.data_paths.processed_attack_data_path) / "og_train_meta_label.npy",
        )
        df_meta_test = load_dataframe(
            Path(cfg.data_paths.processed_attack_data_path),
            "master_challenge_test.csv",
        )
        y_meta_test = np.load(
            Path(cfg.data_paths.processed_attack_data_path) / "master_challenge_test_labels.npy",
        )

        df_synth = load_dataframe(
            Path(cfg.data_paths.processed_attack_data_path),
            "synth.csv",
        )

        df_ref = load_dataframe(
            Path(cfg.data_paths.population_path),
            "population_all_with_challenge_no_id.csv",
        )

        # Fit the metaclassifier.
        # 1. Initialize the attacker
        blending_attacker = BlendingPlusPlus(
            meta_classifier_type=cfg.metaclassifier.model_type, data_configs=cfg.data_configs
        )

        # 2. Train the attacker on the meta-train set
        blending_attacker.fit(
            df_train=df_meta_train,
            y_train=y_meta_train,
            df_synth=df_synth,
            df_ref=df_ref,
            use_gpu=cfg.metaclassifier.use_gpu,
            epochs=cfg.metaclassifier.epochs,
        )

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # model_filename = f"{timestamp}_trained_metaclassifier.pkl"
        # with open(Path(cfg.model_paths.metaclassifier_model_path) / model_filename, "wb") as f:
        #     pickle.dump(blending_attacker.meta_classifier_, f)

        # # 3. Get predictions on the test set
        # final_predictions = blending_attacker.predict(
        #     df_test=df_meta_test,
        #     df_synth=df_synth,
        #     df_ref=df_ref,
        #     cat_cols=cfg.data_configs.metadata.categorical,
        #     y_test=y_meta_test,
        # )

        # print("Final Blending++ predictions:", final_predictions)
        # # TODO: Change print to logging
        # # TODO: Save trained model
        # log(INFO, "Metaclassifier training finished.")


if __name__ == "__main__":
    main()
