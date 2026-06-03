import json
import shutil
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from midst_toolkit.common.enumerations import DataSplit
from midst_toolkit.common.variables import DEVICE
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


def test_train_load_and_synthesize(test_dirs):
    # Ignoring "too many statements" error from Ruff
    # ruff: noqa: PLR0915
    test_data_dir, results_dir = test_dirs
    test_data_name = "trans"
    process_data(test_data_name, test_data_dir, test_data_dir)

    config_file_path = test_data_dir / "config.toml"
    config = load_config(config_file_path)

    # The preprocess function below expects 2 folders of preprocessed data:
    # 1. the dataset to be processed, which can be a subset of the full dataset
    # 2. the full dataset, which is used to unsure it will get all the categories for the categorical features
    # Here we are mocking the dataset #2 (full dataset) by copying the dataset #1
    dataset_path = get_processed_data_dir(test_data_dir) / test_data_name
    ref_dataset_path = Path(f"{dataset_path}_all")
    shutil.copytree(dataset_path, ref_dataset_path)

    # preprocess the data
    numerical_features, categorical_features, categories, d_numerical = preprocess(
        dataset_path=dataset_path,
        ref_dataset_path=ref_dataset_path,
        transforms=config["transforms"],
        task_type=config["task_type"],
    )

    # separate train and test data
    numerical_features_train = numerical_features[DataSplit.TRAIN.value]
    numerical_features_test = numerical_features[DataSplit.TEST.value]
    categorical_features_train = categorical_features[DataSplit.TRAIN.value]
    categorical_features_test = categorical_features[DataSplit.TEST.value]

    # convert to float tensor
    numerical_features_train = torch.tensor(numerical_features_train).float()
    numerical_features_test = torch.tensor(numerical_features_test).float()
    categorical_features_train = torch.tensor(categorical_features_train)
    categorical_features_test = torch.tensor(categorical_features_test)

    # create dataset module
    train_data = TabularDataset(numerical_features_train.float(), categorical_features_train)

    # move test data to gpu if available
    numerical_features_test = numerical_features_test.float().to(DEVICE)
    categorical_features_test = categorical_features_test.to(DEVICE)

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
        numerical_features_test,
        categorical_features_test,
        num_numerical_features=d_numerical,
        num_classes=categories,
        device=DEVICE,
    )

    model_save_dir = results_dir / test_data_name
    vae_save_dir = model_save_dir / "vae"
    vae_save_dir.mkdir(parents=True, exist_ok=True)

    ###### A. Train the VAE model ######

    # instantiate VAE model for training
    tabsyn.instantiate_vae(
        **config["model_params"],
        optim_params=config["train"]["optim"]["vae"],
    )

    tabsyn.train_vae(
        **config["loss_params"],
        num_epochs=config["train"]["vae"]["num_epochs"],
        save_path=vae_save_dir,
    )

    # embed all inputs in the latent space
    tabsyn.save_vae_embeddings(
        numerical_features_train,
        categorical_features_train,
        vae_ckpt_dir=vae_save_dir,
    )
    tabsyn.save_embeddings_attributes(vae_ckpt_dir=vae_save_dir)

    ###### B. Train the Diffusion model ######

    # load latent space embeddings
    train_z, _ = tabsyn.load_latent_embeddings(vae_save_dir)  # train_z dim: B x in_dim

    # normalize embeddings
    latent_train_data = (train_z - train_z.mean(0)) / 2

    # create data loader
    latent_train_loader = DataLoader(
        latent_train_data,
        batch_size=config["train"]["diffusion"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["diffusion"]["num_dataset_workers"],
    )

    # instantiate diffusion model for training
    tabsyn.instantiate_diffusion(
        in_dim=train_z.shape[1],
        optim_params=config["train"]["optim"]["diffusion"],
    )

    # train diffusion model
    tabsyn.train_diffusion(
        latent_train_loader,
        num_epochs=config["train"]["diffusion"]["num_epochs"],
        ckpt_path=model_save_dir,
    )

    ###### Load the model ######

    # instantiate VAE model
    tabsyn.instantiate_vae(**config["model_params"], optim_params=None)

    # load latent embeddings attributes of input data
    train_z_att = tabsyn.load_embeddings_attributes(vae_save_dir)
    token_dim = train_z_att["token_dim"]
    in_dim = train_z_att["in_dim"]

    # instantiate diffusion model
    tabsyn.instantiate_diffusion(in_dim=in_dim, optim_params=None)

    # load state from checkpoint
    tabsyn.load_model_state(ckpt_dir=model_save_dir, dif_ckpt_name="model.pt")

    ###### Synthesize data ######

    # get inverse tokenizers
    _, _, categories, d_numerical, num_inverse, cat_inverse = preprocess(
        dataset_path=dataset_path,
        ref_dataset_path=ref_dataset_path,
        transforms=config["transforms"],
        task_type=config["task_type"],
        inverse=True,
    )

    synthetic_data_dir = results_dir / test_data_name / "synthetic_data"
    synthetic_data_dir.mkdir(parents=True, exist_ok=True)

    # load data info file
    with open(dataset_path / "info.json", "r") as file:
        data_info = json.load(file)

    data_info["token_dim"] = token_dim

    # sample data
    num_samples = train_z_att["num_samples"]
    in_dim = train_z_att["in_dim"]
    mean_input_emb = train_z_att["mean_input_emb"]
    tabsyn.sample(
        num_samples,
        in_dim,
        mean_input_emb,
        info=data_info,
        num_inverse=num_inverse,
        cat_inverse=cat_inverse,
        save_path=synthetic_data_dir / f"{test_data_name}_synthetic.csv",
    )
