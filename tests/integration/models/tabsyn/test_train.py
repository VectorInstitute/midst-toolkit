from pathlib import Path

from midst_toolkit.models.tabsyn.config import load_config
from midst_toolkit.models.tabsyn.dataset import preprocess
from midst_toolkit.models.tabsyn.preprocessing import get_processed_data_dir, process_data


def test_train():
    test_data_name = "trans"
    test_data_name_all = "trans_all"
    test_data_dir = Path("tests/integration/assets/tabsyn")
    process_data(test_data_name, test_data_dir, test_data_dir)

    config_file_path = "tests/integration/assets/tabsyn/config.toml"
    config = load_config(config_file_path)

    import ipdb

    ipdb.set_trace()

    X_num, X_cat, categories, d_numerical = preprocess(
        get_processed_data_dir(test_data_dir) / test_data_name,
        ref_dataset_path=get_processed_data_dir(test_data_dir) / test_data_name_all,
        transforms=config["transforms"],
        task_type=config["task_type"],
    )
