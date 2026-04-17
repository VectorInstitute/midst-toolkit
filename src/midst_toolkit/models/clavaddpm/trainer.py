"""Trainer class for the ClavaDDPM model."""

from collections.abc import Generator, Iterator
from copy import deepcopy
from logging import INFO

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from midst_toolkit.common.logger import log
from midst_toolkit.common.variables import DEVICE
from midst_toolkit.models.clavaddpm.gaussian_multinomial_diffusion import GaussianMultinomialDiffusion


class ClavaDDPMTrainer:
    def __init__(
        self,
        diffusion_model: GaussianMultinomialDiffusion,
        trainer_iterator: Generator[list[Tensor]],
        learning_rate: float,
        weight_decay: float,
        steps: int,
        device: torch.device = DEVICE,
    ):
        """
        Trainer class for the ClavaDDPM model.

        Args:
            diffusion_model: The diffusion model.
            trainer_iterator: The training iterator. It should yield a list of tensors. The first tensor is the input
                tensor and the second tensor is the output tensor.
            learning_rate: The learning rate.
            weight_decay: The weight decay.
            steps: The number of steps to train.
            device: The device to use. Default is midst_toolkit.common.variables.DEVICE.
        """
        self.diffusion_model = diffusion_model
        self.ema_model = deepcopy(self.diffusion_model._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.trainer_iterator = trainer_iterator
        self.steps = steps
        self.initial_learning_rate = learning_rate
        self.optimizer = torch.optim.AdamW(
            self.diffusion_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.device = device
        self.loss_history = pd.DataFrame(columns=["step", "mloss", "gloss", "loss"])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000

    def _anneal_learning_rate(self, step: int) -> None:
        """
        Anneal the learning rate.

        Args:
            step: The current step.
        """
        frac_done = step / self.steps
        lr = self.initial_learning_rate * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _train_step(self, input_tensor: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        """
        Run a single step of the training loop.

        Args:
            input_tensor: The input tensor.
            target: The target tensor.

        Returns:
            A tuple with 2 values:
                - The multi-class loss.
                - The Gaussian loss.
        """
        input_tensor = input_tensor.to(self.device)
        outputs = {"y": target}
        for key, value in outputs.items():
            outputs[key] = value.long().to(self.device)

        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion_model.mixed_loss(input_tensor, outputs)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def train(self) -> None:
        """Run the training loop."""
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        while step < self.steps:
            # TODO: improve this design. If self.steps is larger than self.train_iter,
            # it will lead to a StopIteration error.
            input_tensor, output = next(self.trainer_iterator)
            batch_loss_multi, batch_loss_gauss = self._train_step(input_tensor, output)

            self._anneal_learning_rate(step)

            curr_count += len(input_tensor)
            curr_loss_multi += batch_loss_multi.item() * len(input_tensor)
            curr_loss_gauss += batch_loss_gauss.item() * len(input_tensor)

            # TODO: improve this code, starting by moving it into a function for better readability and modularity.
            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    log(INFO, f"Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}")

                # TODO: switch this for a concat for better code readability
                self.loss_history.loc[len(self.loss_history)] = [
                    step + 1,
                    mloss,
                    gloss,
                    mloss + gloss,
                ]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion_model._denoise_fn.parameters())

            step += 1


def update_ema(
    target_params: Iterator[nn.Parameter],
    source_params: Iterator[nn.Parameter],
    rate: float = 0.999,
) -> None:
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    Args:
        target_params: the target parameter sequence.
        source_params: the source parameter sequence.
        rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        # TODO: is this doing anything at all? The detach functions will create new tensors,
        # so this will not modify the original tensors, and this function does not return anything.
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)
