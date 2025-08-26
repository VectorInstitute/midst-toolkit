from pathlib import Path


BASE_DATA_DIR = Path("midst_toolkit/attacks/black_box_single_table/ensemble_mia/data")

DATA_CONFIG = {
    # Data processing paths and file names
    ## Input directories:
    "midst_data_path": (BASE_DATA_DIR / "midst_data_all_attacks"),  # Used only for reading the data
    ## Output directories:
    "population_path": BASE_DATA_DIR / "population_data",  # Path where the population data is stored
    "processed_attack_data_path": (
        BASE_DATA_DIR / "attack_data"
    ),  # Path where the processed attack real train and evaluation data is stored
    # Attack folder ids under MIDST data attack files
    "folder_ids": {
        "train": list(range(1, 31)),
        "dev": list(range(51, 61)) + list(range(91, 101)),
        "final": list(range(61, 71)) + list(range(101, 111)),
    },
    # File names in MIDST data directories.
    "single_table_train_data_file_name": "train_with_id.csv",
    "multi_table_train_data_file_name": "trans.csv",
    "challenge_data_file_name": "challenge_with_id.csv",
    # Data Config files path
    "trans_domain_file_path": BASE_DATA_DIR / "data_configs/trans_domain.json",
    "dataset_meta_file_path": BASE_DATA_DIR / "data_configs/dataset_meta.json",
    "trans_json_file_path": BASE_DATA_DIR / "data_configs/trans.json",
}

SHADOW_MODELS_DATA_PATH = BASE_DATA_DIR / "shadow_models_data"
SHADOW_MODELS_ARTIFACTS_PATH = Path(
    "midst_toolkit/attacks/black_box_single_table/ensemble_mia/rmia/shadow_models_artifacts"
)


# Random state for reproducing the results
seed = 42
