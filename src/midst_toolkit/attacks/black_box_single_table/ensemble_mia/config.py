from pathlib import Path


# Data processing paths and file names
DATA_FOLDER_PATH = Path("midst_toolkit/attacks/black_box_single_table/ensemble_mia/data")
## Input directories:
MIDST_DATA_PATH = DATA_FOLDER_PATH / "midst_data_all_attacks"  # Used only for reading the data
TABDDPM_BLACK_BOX_PATH = (
    DATA_FOLDER_PATH / "tabddpm_black_box"
)  # # Used for reading the attack specific files and writing the RMIA scores.

## Output directories:
POPULATION_PATH = DATA_FOLDER_PATH / "population_data"  # Path where the population data is stored
PROCESSED_ATTACK_DATA_PATH = (
    DATA_FOLDER_PATH / "attack_data"
)  # Path where the processed attack real train and evaluation data is stored
SHADOW_MODELS_DATA_PATH = DATA_FOLDER_PATH / "shadow_models_data"
SHADOW_MODELS_ARTIFACTS_PATH = Path(
    "midst_toolkit/attacks/black_box_single_table/ensemble_mia/rmia/shadow_models_artifacts"
)

# Attack folder ids under MIDST data attack files
## Train
train_ids = list(range(1, 31))
## Dev
dev_ids = list(range(51, 61)) + list(range(91, 101))
## Final
final_ids = list(range(61, 71)) + list(range(101, 111))

# Data Config files path
TRANS_DOMAIN_FILE_PATH = DATA_FOLDER_PATH / "data_configs/trans_domain.json"
DATASET_META_FILE_PATH = DATA_FOLDER_PATH / "data_configs/dataset_meta.json"
TRANS_JSON_FILE_PATH = DATA_FOLDER_PATH / "data_configs/trans.json"

# Random state for reproducing the results
seed = 42
