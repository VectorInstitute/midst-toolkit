import json
from logging import INFO
from pathlib import Path

import hydra
from omegaconf import DictConfig

from examples.tabsyn.train import train_tabsyn
from midst_toolkit.common.logger import log
from midst_toolkit.common.variables import DEVICE
from midst_toolkit.models.tabsyn.config import load_config
from midst_toolkit.models.tabsyn.dataset import preprocess
from midst_toolkit.models.tabsyn.pipeline import TabSyn
from midst_toolkit.models.tabsyn.preprocessing import get_processed_data_dir


@hydra.main(config_path=".", config_name="config", version_base=None)
def tabsyn_synthesize(config: DictConfig) -> None:
    """
    Synthesize data using the TabSyn model.

    Args:
        config: Configuration as an OmegaConf DictConfig object.
    """
    log(INFO, "Synthesizing data using the TabSyn model...")

    results_dir = Path(config.results_dir)
    model_save_dir = results_dir / config.table_name
    vae_save_dir = model_save_dir / "vae"

    if not (model_save_dir / "model.pt").exists() or not (vae_save_dir / "model.pt").exists():
        log(INFO, "Trained model not found. Training the model...")
        train_tabsyn(config)

    else:
        log(INFO, "Trained model found.")

    # The preprocess function below expects 2 folders of preprocessed data:
    # 1. the dataset to be processed, which can be a subset of the full dataset
    # 2. the full dataset, which is used to unsure it will get all the categories for the categorical features
    # Here we are making the dataset #2 (full dataset) by copying the dataset #1
    dataset_path = get_processed_data_dir(Path(config.data_dir)) / config.table_name
    ref_dataset_path = Path(f"{dataset_path}_all")

    tabsyn_config = load_config(Path(config.tabsyn_config))

    log(INFO, "Instantiating the TabSyn model...")

    _, _, categories, d_numerical = preprocess(  # type: ignore[misc]
        dataset_path=dataset_path,
        ref_dataset_path=ref_dataset_path,
        transforms=tabsyn_config["transforms"],
        task_type=tabsyn_config["task_type"],
    )

    # Instantiate an empty model object so we can load the model state from the checkpoint
    # TODO: Refactor this constructor to allow for this use case without having to ignore the type checker
    tabsyn = TabSyn(None, None, None, d_numerical, categories, device=DEVICE)  # type: ignore[arg-type]

    ###### Load the model ######

    # instantiate VAE model
    tabsyn.instantiate_vae(**tabsyn_config["model_params"], optim_params=None)

    # load latent embeddings attributes of input data
    train_z_att = tabsyn.load_embeddings_attributes(vae_save_dir)
    token_dim = train_z_att["token_dim"]
    in_dim = train_z_att["in_dim"]
    hid_dim = train_z_att["hid_dim"]

    # instantiate diffusion model
    tabsyn.instantiate_diffusion(in_dim=in_dim, hid_dim=hid_dim, optim_params=None)

    # load state from checkpoint
    tabsyn.load_model_state(ckpt_dir=model_save_dir, dif_ckpt_name="model.pt")

    ###### Synthesize data ######

    # get inverse tokenizers
    # TODO: refactor the return of the preprocess function so we don't need to ignore mypy here
    _, _, _, _, num_inverse, cat_inverse = preprocess(  # type: ignore[misc]
        dataset_path=dataset_path,
        ref_dataset_path=ref_dataset_path,
        transforms=tabsyn_config["transforms"],
        task_type=tabsyn_config["task_type"],
        inverse=True,
    )

    synthetic_data_dir = results_dir / config.table_name / "synthetic_data"
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
        save_path=synthetic_data_dir / f"{config.table_name}_synthetic.csv",
    )

    log(INFO, "Synthesizing Done!")


if __name__ == "__main__":
    tabsyn_synthesize()
