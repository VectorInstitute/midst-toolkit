from collections.abc import Generator, Iterator
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from midst_toolkit.models.clavaddpm.gaussian_multinomial_diffusion import GaussianMultinomialDiffusion


class Trainer:
    def __init__(
        self,
        diffusion: GaussianMultinomialDiffusion,
        train_iter: Generator[tuple[Tensor, ...]],
        lr: float,
        weight_decay: float,
        steps: int,
        device: str = "cuda",
    ):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=["step", "mloss", "gloss", "loss"])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000

    def _anneal_lr(self, step: int) -> None:
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x: Tensor, out_dict: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        x = x.to(self.device)
        for k, v in out_dict.items():
            out_dict[k] = v.long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()  # type: ignore[no-untyped-call]
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self) -> None:
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        while step < self.steps:
            x, out = next(self.train_iter)
            out_dict = {"y": out}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(f"Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}")
                self.loss_history.loc[len(self.loss_history)] = [
                    step + 1,
                    mloss,
                    gloss,
                    mloss + gloss,
                ]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1


def update_ema(
    target_params: Iterator[nn.Parameter],
    source_params: Iterator[nn.Parameter],
    rate: float = 0.999,
) -> None:
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)
