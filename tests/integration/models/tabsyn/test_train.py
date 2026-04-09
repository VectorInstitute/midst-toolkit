import shutil
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from midst_toolkit.models.tabsyn.config import load_config
from midst_toolkit.models.tabsyn.dataset import TabularDataset, preprocess
from midst_toolkit.models.tabsyn.pipeline import TabSyn
from midst_toolkit.models.tabsyn.preprocessing import get_processed_data_dir, process_data


@pytest.fixture
def test_dirs():
    test_data_dir = Path("tests/integration/assets/tabsyn")
    results_dir = test_data_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    yield test_data_dir, results_dir

    shutil.rmtree(get_processed_data_dir(test_data_dir), ignore_errors=True)
    shutil.rmtree(results_dir, ignore_errors=True)


def test_train(test_dirs):
    test_data_dir, results_dir = test_dirs
    test_data_name = "trans"
    process_data(test_data_name, test_data_dir, test_data_dir)

    config_file_path = "tests/integration/assets/tabsyn/config.toml"
    config = load_config(config_file_path)

    # The preprocess function below expects 2 folders of preprocessed data:
    # 1. the dataset to be processed, which can be a subset of the full dataset
    # 2. the full dataset, which is used to unsure it will get all the categories for the categorical features
    # Here we are mocking the dataset #2 (full dataset) by copying the dataset #1
    dataset_path = get_processed_data_dir(test_data_dir) / test_data_name
    ref_dataset_path = f"{dataset_path}_all"
    shutil.copytree(dataset_path, ref_dataset_path)

    # preprocess the data
    X_num, X_cat, categories, d_numerical = preprocess(
        dataset_path=dataset_path,
        ref_dataset_path=ref_dataset_path,
        transforms=config["transforms"],
        task_type=config["task_type"],
    )

    # separate train and test data
    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat

    # convert to float tensor
    X_train_num, X_test_num = (
        torch.tensor(X_train_num).float(),
        torch.tensor(X_test_num).float(),
    )
    X_train_cat, X_test_cat = torch.tensor(X_train_cat), torch.tensor(X_test_cat)

    # create dataset module
    train_data = TabularDataset(X_train_num.float(), X_train_cat)

    # move test data to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_test_num = X_test_num.float().to(device)
    X_test_cat = X_test_cat.to(device)

    # create train dataloader
    train_loader = DataLoader(
        train_data,
        batch_size=config["train"]["vae"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["vae"]["num_dataset_workers"],
    )

    # Instantiate the model
    tabsyn = TabSyn(
        train_loader,
        X_test_num,
        X_test_cat,
        num_numerical_features=d_numerical,
        num_classes=categories,
        device=device,
    )

    # instantiate VAE model for training
    tabsyn.instantiate_vae(
        **config["model_params"],
        optim_params=config["train"]["optim"]["vae"],
    )

    tabsyn.train_vae(
        **config["loss_params"],
        num_epochs=config["train"]["vae"]["num_epochs"],
        save_path=results_dir / test_data_name / "vae",
    )
