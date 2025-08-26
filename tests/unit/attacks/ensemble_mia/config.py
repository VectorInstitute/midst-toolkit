from pathlib import Path

BASE_DATA_DIR = Path("tests/unit/attacks/ensemble_mia/assets")

DATA_CONFIG = {
    # Data processing paths and file names
    ## Input directories:
    "midst_data_path": BASE_DATA_DIR / "midst_data_all_attacks",  # Used only for reading the data
    ## Output directories:
    "population_path": BASE_DATA_DIR
    / "population_data",  # Path where the population data is stored
    "processed_attack_data_path": (
        BASE_DATA_DIR / "attack_data"
    ),  # Path where the processed attack real train and evaluation data is stored
    # Attack folder ids under MIDST data attack files
    "folder_ids": {
        "train": [1, 2],
        "dev": [51, 52],
        "final": [61, 62],
    },
    # File names in MIDST data directories.
    "single_table_train_data_file_name": "train_with_id.csv",
    "multi_table_train_data_file_name": "trans.csv",
    "challenge_data_file_name": "challenge_with_id.csv",
}
