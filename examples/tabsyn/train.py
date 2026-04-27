import shutil
from logging import INFO
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader

from midst_toolkit.common.enumerations import DataSplit
from midst_toolkit.common.logger import log
from midst_toolkit.common.variables import DEVICE
from midst_toolkit.models.tabsyn.config import load_config
from midst_toolkit.models.tabsyn.dataset import TabularDataset, preprocess
from midst_toolkit.models.tabsyn.pipeline import TabSyn
from midst_toolkit.models.tabsyn.preprocessing import get_processed_data_dir, process_data


@hydra.main(config_path=".", config_name="config", version_base=None)
def train_tabsyn(config: DictConfig) -> None:
    """
    Train a TabSyn model.

    Args:
        config: Configuration as an OmegaConf DictConfig object.
    """
    log(INFO, "Training TabSyn model...")
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    process_data(config.table_name, Path(config.data_dir), Path(config.data_dir))

    current_dir = Path(__file__).resolve().parent
    tabsyn_config = load_config(current_dir / "tabsyn_config.toml")

    # The preprocess function below expects 2 folders of preprocessed data:
    # 1. the dataset to be processed, which can be a subset of the full dataset
    # 2. the full dataset, which is used to unsure it will get all the categories for the categorical features
    # Here we are making the dataset #2 (full dataset) by copying the dataset #1
    dataset_path = get_processed_data_dir(Path(config.data_dir)) / config.table_name
    ref_dataset_path = Path(f"{dataset_path}_all")
    shutil.rmtree(ref_dataset_path, ignore_errors=True)
    shutil.copytree(dataset_path, ref_dataset_path)

    log(INFO, "Preprocessing data...")

    # preprocess the data
    # TODO: refactor the return of the preprocess function so we don't need to ignore mypy here
    numerical_features, categorical_features, categories, d_numerical = preprocess(  # type: ignore[misc]
        dataset_path=dataset_path,
        ref_dataset_path=ref_dataset_path,
        transforms=tabsyn_config["transforms"],
        task_type=tabsyn_config["task_type"],
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

    log(INFO, "Loading the dataset...")

    # create dataset module
    train_data = TabularDataset(numerical_features_train.float(), categorical_features_train)

    # move test data to gpu if available
    numerical_features_test = numerical_features_test.float().to(DEVICE)
    categorical_features_test = categorical_features_test.to(DEVICE)

    # create train dataloader
    train_loader: DataLoader[TabularDataset] = DataLoader[TabularDataset](
        # Ignoring here because this is expecting the dataset to be subclass of torch's Dataset but it isn't
        train_data,  # type: ignore[arg-type]
        batch_size=tabsyn_config["train"]["vae"]["batch_size"],
        shuffle=True,
        num_workers=tabsyn_config["train"]["vae"]["num_dataset_workers"],
    )

    log(INFO, "Instantiating the TabSyn model...")

    # Instantiate the model
    tabsyn = TabSyn(
        train_loader,
        numerical_features_test,
        categorical_features_test,
        num_numerical_features=d_numerical,
        num_classes=categories,
        device=DEVICE,
    )

    model_save_dir = results_dir / config.table_name
    vae_save_dir = model_save_dir / "vae"
    vae_save_dir.mkdir(parents=True, exist_ok=True)

    ###### A. Train the VAE model ######

    log(INFO, "Training the TabSyn VAE model...")

    # instantiate VAE model for training
    tabsyn.instantiate_vae(
        **tabsyn_config["model_params"],
        optim_params=tabsyn_config["train"]["optim"]["vae"],
    )

    tabsyn.train_vae(
        **tabsyn_config["loss_params"],
        num_epochs=tabsyn_config["train"]["vae"]["num_epochs"],
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

    log(INFO, "Training the TabSyn Diffusion model...")

    # load latent space embeddings
    train_z, _ = tabsyn.load_latent_embeddings(vae_save_dir)  # train_z dim: B x in_dim

    # normalize embeddings
    latent_train_data = (train_z - train_z.mean(0)) / 2

    # create data loader
    latent_train_loader: DataLoader[Tensor] = DataLoader[Tensor](
        # Ignoring the type checker here because our code in tabsyn.train_diffusion
        # works with plain Tensor and not with TensorDataset
        latent_train_data,  # type: ignore[arg-type]
        batch_size=tabsyn_config["train"]["diffusion"]["batch_size"],
        shuffle=True,
        num_workers=tabsyn_config["train"]["diffusion"]["num_dataset_workers"],
    )

    # instantiate diffusion model for training
    tabsyn.instantiate_diffusion(
        in_dim=train_z.shape[1],
        hid_dim=train_z.shape[1],
        optim_params=tabsyn_config["train"]["optim"]["diffusion"],
    )

    # train diffusion model
    tabsyn.train_diffusion(
        latent_train_loader,
        num_epochs=tabsyn_config["train"]["diffusion"]["num_epochs"],
        ckpt_path=model_save_dir,
    )

    log(INFO, "Done!")


if __name__ == "__main__":
    train_tabsyn()
