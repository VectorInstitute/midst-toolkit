import os
import pickle
import time
from collections.abc import Callable
from logging import INFO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from midst_toolkit.common.logger import log
from midst_toolkit.common.variables import DEVICE
from midst_toolkit.models.tabsyn.model.modules import (
    MLPDiffusion,
    Model,
)
from midst_toolkit.models.tabsyn.model.utils import sample
from midst_toolkit.models.tabsyn.model.vae import (
    DecoderModel,
    EncoderModel,
    ModelVAE,
)
from midst_toolkit.models.tabsyn.utils import (
    recover_data,
    split_num_cat_target,
)


class TabSyn:
    def __init__(
        self,
        train_loader: DataLoader,
        numerical_features_test: Tensor,
        categorical_features_test: Tensor,
        num_numerical_features: int,
        num_classes: list[int],
        device: torch.device = DEVICE,
    ) -> None:
        """
        Init the TabSyn pipeline for training and sampling.

        Args:
            train_loader: The loader for the training data.
            numerical_features_test: The numerical features of the test data.
            categorical_features_test: The categorical features of the test data.
            num_numerical_features: The number of numerical features.
            num_classes: The number of classes for each categorical feature.
            device: The device to use for the model. Optional, defaults to midst_toolkit.common.variables.DEVICE.
        """
        self.train_loader = train_loader
        self.numerical_features_test = numerical_features_test
        self.categorical_features_test = categorical_features_test
        self.d_numerical = num_numerical_features
        self.categories = num_classes
        self.device = device

    def instantiate_vae(
        self,
        n_head: int,
        factor: int,
        num_layers: int,
        d_token: int,
        optim_params: dict[str, Any] | None = None,
    ) -> None:
        """Construct VAE model and its optimizer and lr scheduler.

        Args:
            n_head: The number of heads.
            factor: The factor for the dimension of the hidden layer.
            num_layers: The number of layers.
            d_token: The dimension of the token.
            optim_params: The optimizer parameters. Optional, defaults to None.
        """
        # construct vae model
        self.vae_model, self.pre_encoder, self.pre_decoder = self.__get_vae_model(n_head, factor, num_layers, d_token)
        # construct vae optimizer and scheduler
        if optim_params is not None:
            self.vae_optimizer, self.vae_scheduler = self.__load_optim(self.vae_model, **optim_params)
        log(INFO, "Successfully instantiated VAE model.")

    def instantiate_diffusion(self, in_dim: int, optim_params: dict[str, Any] | None) -> None:
        """Construct Diffusion model and its optimizer and lr scheduler.

        Args:
            in_dim: The dimension of the input.
            optim_params: The optimizer parameters. Optional, defaults to None.
        """
        # load diffusion model
        self.diffusion_model = self.__get_diffusion_model(input_dimension=in_dim)
        # load optimizer and scheduler
        if optim_params is not None:
            self.dif_optimizer, self.dif_scheduler = self.__load_optim(self.diffusion_model, **optim_params)
        log(INFO, "Successfully instantiated diffusion model.")

    def __get_vae_model(
        self,
        n_head: int,
        factor: int,
        num_layers: int,
        d_token: int,
    ) -> tuple[ModelVAE, EncoderModel, DecoderModel]:
        model = ModelVAE(
            num_layers,
            self.d_numerical,
            self.categories,
            d_token,
            n_head=n_head,
            factor=factor,
            bias=True,
        )
        model = model.to(self.device)

        pre_encoder = EncoderModel(
            num_layers,
            self.d_numerical,
            self.categories,
            d_token,
            n_head=n_head,
            factor=factor,
        ).to(self.device)
        pre_decoder = DecoderModel(
            num_layers,
            self.d_numerical,
            self.categories,
            d_token,
            n_head=n_head,
            factor=factor,
        ).to(self.device)

        pre_encoder.eval()
        pre_decoder.eval()

        return model, pre_encoder, pre_decoder

    def __get_diffusion_model(self, input_dimension: int) -> Model:
        denoise_model = MLPDiffusion(input_dimension, 1024).to(self.device)
        log(INFO, f"Denoise model: {denoise_model}")

        num_params = sum(p.numel() for p in denoise_model.parameters())
        log(INFO, f"The number of parameters: {num_params}")

        return Model(denoise_model=denoise_model).to(self.device)

    def __load_optim(
        self, model: nn.Module, lr: float, weight_decay: float, factor: float, patience: int
    ) -> tuple[torch.optim.Optimizer, ReduceLROnPlateau]:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience)
        return optimizer, scheduler

    # TODO: Refactor this to get rid of the "too many statements" Ruff error
    def train_vae(self, max_beta: float, min_beta: float, lambd: float, num_epochs: int, save_path: Path) -> None:
        # ruff: noqa: PLR0915
        """Train the VAE model.

        Args:
            max_beta: The maximum beta.
            min_beta: The minimum beta.
            lambd: The lambda for the beta.
            num_epochs: The number of epochs to train the model.
            save_path: The path to save the model.
        """
        # determine model save paths
        model_save_path = save_path / "model.pt"
        encoder_save_path = save_path / "encoder.pt"
        decoder_save_path = save_path / "decoder.pt"

        # set initial state
        current_lr = self.vae_optimizer.param_groups[0]["lr"]
        patience = 0
        best_train_loss = float("inf")
        beta = max_beta

        # training loop
        start_time = time.time()
        for epoch in range(num_epochs):
            pbar = tqdm(self.train_loader, total=len(self.train_loader))
            pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")

            curr_loss_multi = 0.0
            curr_loss_gauss = 0.0
            curr_loss_kl = 0.0

            curr_count = 0

            for batch_num, batch_cat in pbar:
                self.vae_model.train()
                self.vae_optimizer.zero_grad()

                batch_num = batch_num.to(self.device)
                batch_cat = batch_cat.to(self.device)

                reconstructed_numerical_features, reconstructed_categorical_features, mu_z, std_z = self.vae_model(
                    batch_num, batch_cat
                )

                loss_mse, loss_ce, loss_kld, train_acc = self.compute_loss(
                    batch_num,
                    batch_cat,
                    reconstructed_numerical_features,
                    reconstructed_categorical_features,
                    mu_z,
                    std_z,
                )

                loss = loss_mse + loss_ce + beta * loss_kld
                loss.backward()
                self.vae_optimizer.step()

                batch_length = batch_num.shape[0]
                curr_count += batch_length
                curr_loss_multi += loss_ce.item() * batch_length
                curr_loss_gauss += loss_mse.item() * batch_length
                curr_loss_kl += loss_kld.item() * batch_length

            num_loss = curr_loss_gauss / curr_count
            cat_loss = curr_loss_multi / curr_count
            kl_loss = curr_loss_kl / curr_count

            """
                Evaluation
            """
            self.vae_model.eval()
            with torch.no_grad():
                reconstructed_numerical_features, reconstructed_categorical_features, mu_z, std_z = self.vae_model(
                    self.numerical_features_test, self.categorical_features_test
                )

                val_mse_loss, val_ce_loss, _, val_acc = self.compute_loss(
                    self.numerical_features_test,
                    self.categorical_features_test,
                    reconstructed_numerical_features,
                    reconstructed_categorical_features,
                    mu_z,
                    std_z,
                )
                val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()

                self.vae_scheduler.step(val_loss)
                new_lr = self.vae_optimizer.param_groups[0]["lr"]

                if new_lr != current_lr:
                    current_lr = new_lr
                    log(INFO, f"Learning rate updated: {current_lr}")

                train_loss = val_loss
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    patience = 0
                    torch.save(self.vae_model.state_dict(), model_save_path)
                else:
                    patience += 1
                    if patience == 10 and beta > min_beta:
                        beta = beta * lambd

            log(
                INFO,
                "epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}".format(
                    epoch,
                    beta,
                    num_loss,
                    cat_loss,
                    kl_loss,
                    val_mse_loss.item(),
                    val_ce_loss.item(),
                    train_acc.item(),
                    val_acc.item(),
                ),
            )

        end_time = time.time()
        log(INFO, f"Training time: {((end_time - start_time) / 60):.4f} mins")

        # load and save encoder and decoder states
        self.pre_encoder.load_weights(self.vae_model)
        self.pre_decoder.load_weights(self.vae_model)

        torch.save(self.pre_encoder.state_dict(), encoder_save_path)
        torch.save(self.pre_decoder.state_dict(), decoder_save_path)

        log(INFO, "Successfully trained and saved the VAE model!")

    def compute_loss(
        self,
        numerical_features: Tensor,
        categorical_features: Tensor,
        reconstructed_numerical_features: Tensor,
        reconstructed_categorical_features: Tensor,
        mu_z: Tensor,
        logvar_z: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute the loss for the VAE.

        Args:
            numerical_features: The numerical features.
            categorical_features: The categorical features.
            reconstructed_numerical_features: The reconstructed numerical features.
            reconstructed_categorical_features: The reconstructed categorical features.
            mu_z: The mean of the latent variables.
            logvar_z: The log variance of the latent variables.

        Returns:
            A tuple containing:
            - The MSE loss.
            - The CE loss.
            - The KL divergence loss.
            - The accuracy.
        """
        ce_loss_fn = nn.CrossEntropyLoss()
        mse_loss = (numerical_features - reconstructed_numerical_features).pow(2).mean()
        ce_loss = torch.tensor(0.0, device=self.device)
        acc = torch.tensor(0.0, device=self.device)
        total_num = torch.tensor(0.0, device=self.device)

        index = 0
        for x_cat in reconstructed_categorical_features:
            if x_cat is not None:
                ce_loss += ce_loss_fn(x_cat, categorical_features[:, index])
                x_hat = x_cat.argmax(dim=-1)
                total_num += x_hat.shape[0]

            acc += (x_hat == categorical_features[:, index]).float().sum()
            index += 1

        if index > 0:
            ce_loss /= index

        acc /= total_num
        # loss = mse_loss + ce_loss

        temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

        loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
        return mse_loss, ce_loss, loss_kld, acc

    def save_vae_embeddings(
        self,
        numerical_features_train: Tensor,
        categorical_features_train: Tensor,
        vae_ckpt_dir: Path,
    ) -> None:
        """Save the latent embeddings to the checkpoint directory.

        Args:
            numerical_features_train: The numerical features of the training data.
            categorical_features_train: The categorical features of the training data.
            vae_ckpt_dir: The path to the checkpoint directory.
        """
        # Saving latent embeddings
        with torch.no_grad():
            numerical_features_train = numerical_features_train.to(self.device)
            categorical_features_train = categorical_features_train.to(self.device)

            train_z = self.pre_encoder(numerical_features_train, categorical_features_train).detach().cpu().numpy()

            np.save(vae_ckpt_dir / "train_z.npy", train_z)

            log(INFO, "Successfully saved pretrained embeddings on disk!")

    def load_latent_embeddings(self, vae_ckpt_dir: Path) -> tuple[Tensor, int]:
        """Load the latent embeddings from the checkpoint directory.

        Args:
            vae_ckpt_dir: The path to the checkpoint directory.

        Returns:
            A tuple containing the latent embeddings and the token dimension.
        """
        embedding_save_path = vae_ckpt_dir / "train_z.npy"
        train_z = torch.tensor(np.load(embedding_save_path)).float()

        # flatten embeddings
        train_z = train_z[:, 1:, :]
        num_samples, num_tokens, token_dim = train_z.size()
        in_dim = num_tokens * token_dim

        train_z = train_z.view(num_samples, in_dim)

        return train_z, token_dim

    def save_embeddings_attributes(self, vae_ckpt_dir: Path) -> None:
        """Save the embeddings attributes to the checkpoint directory.

        Args:
            vae_ckpt_dir: The path to the checkpoint directory.
        """
        train_z, token_dim = self.load_latent_embeddings(vae_ckpt_dir)
        embedding_att = {
            "token_dim": token_dim,
            "in_dim": train_z.shape[1],
            "hid_dim": train_z.shape[1],
            "num_samples": train_z.shape[0],
            "mean_input_emb": train_z.mean(0),
            "std_input_emb": train_z.std(0),
        }
        with open(vae_ckpt_dir / "train_z_attributes.pkl", "wb") as file:
            pickle.dump(embedding_att, file)

    def load_embeddings_attributes(self, vae_ckpt_dir: Path) -> dict[str, Any]:
        """Load the embeddings attributes from the checkpoint directory.

        Args:
            vae_ckpt_dir: The path to the checkpoint directory.

        Returns:
            The embeddings attributes as a dictionary.
        """
        return pickle.load(open(os.path.join(vae_ckpt_dir, "train_z_attributes.pkl"), "rb"))

    def train_diffusion(self, train_loader: DataLoader, num_epochs: int, ckpt_path: Path) -> None:
        """Train the diffusion model.

        Args:
            train_loader: The loader for the training data.
            num_epochs: The number of epochs to train the model.
            ckpt_path: The path to save the checkpoint.
        """
        self.diffusion_model.train()

        best_loss = float("inf")
        patience = 0
        start_time = time.time()
        for epoch in range(num_epochs):
            pbar = tqdm(train_loader, total=len(train_loader))
            pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")

            batch_loss = 0.0
            len_input = 0
            for batch in pbar:
                inputs = batch.float().to(self.device)
                loss = self.diffusion_model(inputs)

                loss = loss.mean()

                batch_loss += loss.item() * len(inputs)
                len_input += len(inputs)

                self.dif_optimizer.zero_grad()
                loss.backward()
                self.dif_optimizer.step()

                pbar.set_postfix({"Loss": loss.item()})

            curr_loss = batch_loss / len_input
            self.dif_scheduler.step(curr_loss)

            if curr_loss < best_loss:
                best_loss = curr_loss
                patience = 0
                torch.save(self.diffusion_model.state_dict(), ckpt_path / "model.pt")
            else:
                patience += 1
                if patience == 500:
                    log(INFO, "Early stopping")
                    break

            if epoch % 1000 == 0:
                torch.save(
                    self.diffusion_model.state_dict(),
                    ckpt_path / f"model_{epoch}.pt",
                )

        end_time = time.time()
        log(INFO, f"Time: {end_time - start_time}")

    def load_model_state(self, ckpt_dir: Path, dif_ckpt_name: str = "model.pt") -> None:
        """Load the model state from the checkpoint directory.

        Args:
            ckpt_dir: The path to the checkpoint directory.
            dif_ckpt_name: The name of the diffusion model checkpoint.
        """
        dif_model_save_path = ckpt_dir / dif_ckpt_name
        vae_model_save_path = ckpt_dir / "vae" / "model.pt"
        encoder_save_path = ckpt_dir / "vae" / "encoder.pt"
        decoder_save_path = ckpt_dir / "vae" / "decoder.pt"

        self.diffusion_model.load_state_dict(torch.load(dif_model_save_path, map_location=self.device))
        self.vae_model.load_state_dict(torch.load(vae_model_save_path, map_location=self.device))
        self.pre_encoder.load_state_dict(torch.load(encoder_save_path, map_location=self.device))
        self.pre_decoder.load_state_dict(torch.load(decoder_save_path, map_location=self.device))

        log(INFO, f"Loaded model state from {ckpt_dir}")

    def load_model_for_sampling(
        self,
        in_dim: int,
        d_numerical: int,
        categories: list[int],
        ckpt_dir: Path,
        n_head: int,
        factor: int,
        num_layers: int,
        d_token: int,
    ) -> None:
        """Load the model for sampling.

        Args:
            in_dim: The dimension of the input.
            d_numerical: The number of numerical features.
            categories: The number of categories for each categorical feature.
            ckpt_dir: The path to the checkpoint directory.
            n_head: The number of heads.
            factor: The factor for the dimension of the hidden layer.
            num_layers: The number of layers.
            d_token: The dimension of the token.
        """
        denoise_model = MLPDiffusion(in_dim, 1024).to(self.device)
        model = Model(denoise_model=denoise_model).to(self.device)
        model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=self.device))

        pre_decoder = DecoderModel(num_layers, d_numerical, categories, d_token, n_head=n_head, factor=factor)
        decoder_save_path = ckpt_dir / "vae" / "decoder.pt"
        pre_decoder.load_state_dict(torch.load(decoder_save_path, map_location=self.device))

        self.diffusion_model = model
        self.pre_decoder = pre_decoder

    def sample(
        self,
        num_samples: int,
        in_dim: int,
        mean_input_emb: Tensor,
        info: dict[str, Any],
        num_inverse: Callable,
        cat_inverse: Callable,
        save_path: Path,
    ) -> None:
        """
        Generate samples from the diffusion model.

        Args:
            num_samples: The number of samples to generate.
            in_dim: The dimension of the input.
            mean_input_emb: The mean of the input.
            info: The information dictionary about the columns.
            num_inverse: The inverse function for the numerical features.
            cat_inverse: The inverse function for the categorical features.
            save_path: The path to save the generated samples.
        """
        self.pre_decoder.cpu()
        info["pre_decoder"] = self.pre_decoder

        mean = mean_input_emb

        start_time = time.time()

        sample_dim = in_dim

        x_next = sample(self.diffusion_model.preconditioner, num_samples, sample_dim)
        x_next = x_next * 2 + mean.to(self.device)

        syn_data = x_next.float().cpu().numpy()
        syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse)

        syn_df = recover_data(syn_num, syn_cat, syn_target, info)

        idx_name_mapping = info["idx_name_mapping"]
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

        syn_df.rename(columns=idx_name_mapping, inplace=True)
        syn_df.to_csv(save_path, index=False)

        end_time = time.time()
        log(INFO, f"Time: {end_time - start_time}")

        self.pre_decoder.to(self.device)

        log(INFO, f"Saving sampled data to {save_path}")
