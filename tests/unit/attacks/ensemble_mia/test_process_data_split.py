from pathlib import Path
from src.midst_toolkit.attacks.black_box_single_table.ensemble_mia.data_processing.process_split_data import (
    process_split_data,
)
from tests.unit.attacks.ensemble_mia.config import DATA_CONFIG

from src.midst_toolkit.attacks.black_box_single_table.ensemble_mia.data_processing.data_utils import load_dataframe

def test_process_split_data(tmp_path: Path) -> None:
    # Comment the next line to update processed attack data stored in DATA_CONFIG["processed_attack_data_path"].
    DATA_CONFIG["processed_attack_data_path"] = tmp_path

    process_split_data(data_config=DATA_CONFIG)

    # Assert that the split real data files are saved in the provided path
    assert (DATA_CONFIG["processed_attack_data_path"] / "real_train.csv").exists()
    assert (DATA_CONFIG["processed_attack_data_path"] / "real_val.csv").exists()
    assert (DATA_CONFIG["processed_attack_data_path"] / "real_test.csv").exists()
    assert (DATA_CONFIG["processed_attack_data_path"] / "real_test.csv").exists()

    # Assert that the master challenge data files are saved in the provided path
    assert (DATA_CONFIG["processed_attack_data_path"] / "master_challenge_train.csv").exists()
    assert (DATA_CONFIG["processed_attack_data_path"] / "master_challenge_train_labels.npy").exists()
    assert (DATA_CONFIG["processed_attack_data_path"] / "master_challenge_test.csv").exists()
    assert (DATA_CONFIG["processed_attack_data_path"] / "master_challenge_test_labels.npy").exists()

    # Assert that the collected data has the expected number of rows and columns
    real_train = load_dataframe(DATA_CONFIG["processed_attack_data_path"], "real_train.csv")
    # The whole real data is 40k samples, half of which is assigned to training set.
    assert real_train.shape == (20000, 10), f"Shape is {real_train.shape}"

    # Real val and test sets each contain a quarter of the whole real data (40k samples).
    real_val = load_dataframe(DATA_CONFIG["processed_attack_data_path"], "real_val.csv")
    assert real_val.shape == (10000, 10), f"Shape is {real_val.shape}"
    real_test = load_dataframe(DATA_CONFIG["processed_attack_data_path"], "real_test.csv")
    assert real_test.shape == (10000, 10), f" Shape is {real_test.shape}"

    # Recall that `master_challenge_train`` consists of two halves: one half (10k) from `real_val`` data
    # with their "is_train" column set to 0, and the other half (10k) from the real train data (`real_train``)
    # with their "is_train" column set to 1. Note that ["is_train"] column is dropped in the final dataframes.
    master_challenge_train = load_dataframe(
        DATA_CONFIG["processed_attack_data_path"], "master_challenge_train.csv"
    )
    assert master_challenge_train.shape == (20000, 10), f" Shape is {master_challenge_train.shape}"

    # Recall that `master_challenge_test`` consists of two halves: one half (10k) from `real_test`` data
    # with their "is_train" column set to 0, and the other half (10k) from the real train data (`real_train``)
    # with their "is_train" column set to 1. Note that ["is_train"] column is dropped in the final dataframes.
    master_challenge_test = load_dataframe(
        DATA_CONFIG["processed_attack_data_path"], "master_challenge_test.csv"
    )
    assert master_challenge_test.shape == (20000, 10), f" Shape is {master_challenge_test.shape}"
