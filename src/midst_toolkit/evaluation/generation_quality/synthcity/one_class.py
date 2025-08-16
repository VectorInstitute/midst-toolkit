# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# third party
from logging import DEBUG

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from midst_toolkit.common.logger import log
from midst_toolkit.evaluation.generation_quality.synthcity.networks import build_network


def one_class_loss(outputs: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Computes the sum of the Euclidean distances from the center tensor (c) and then the mean over the sum.

    Args:
        outputs: Output from the neural network for the batch.
        c: center point, from which we're measuring the Euclidean distances.

    Returns:
        Mean distances.
    """
    dist = torch.sum((outputs - c) ** 2, dim=1)
    return torch.mean(dist)


def soft_boundary_loss(outputs: torch.Tensor, r: torch.Tensor, c: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """A similar loss function to the one class loss but with some small modifications."""
    dist = torch.sum((outputs - c) ** 2, dim=1)
    scores = dist
    return (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))


class BaseNet(nn.Module):
    def __init__(self) -> None:
        """Base class for all neural networks."""
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Abstract forward pass through the network.

        Args:
            X: input to the network

        Raises:
            NotImplementedError: Must be implemented by the inheriting network

        Returns:
            Output of the network
        """
        raise NotImplementedError


def get_radius(dist: torch.Tensor, nu: float) -> np.ndarray:
    """
    Optimally solve for radius R via the (1-nu)-quantile of distances.

    Args:
        dist: Distances tensor
        nu: hyper-parameter for the quantile

    Returns:
        Radii
    """
    return np.quantile(np.sqrt(dist.clone().data.float().cpu().numpy()), 1 - nu)


class OneClassLayer(BaseNet):
    def __init__(
        self,
        input_dim: int,
        rep_dim: int,
        center: torch.Tensor,
        num_layers: int = 4,
        num_hidden: int = 32,
        activation: str = "ReLU",
        dropout_prob: float = 0.2,
        dropout_active: bool = False,
        lr: float = 2e-3,
        epochs: int = 1000,
        warm_up_epochs: int = 20,
        train_prop: float = 1.0,
        weight_decay: float = 2e-3,
        radius: float = 1,
        nu: float = 1e-2,
    ):
        """Neural network."""
        super().__init__()

        self.rep_dim = rep_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.activation = activation
        self.dropout_prob = dropout_prob
        self.dropout_active = dropout_active
        self.train_prop = train_prop
        self.learningRate = lr
        self.epochs = epochs
        self.warm_up_epochs = warm_up_epochs
        self.weight_decay = weight_decay
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # Make this an option
        else:
            self.device = torch.device("cpu")
        # set up the network

        self.model = build_network(
            network_name="feedforward",
            params={
                "input_dim": input_dim,
                "rep_dim": rep_dim,
                "num_hidden": num_hidden,
                "activation": activation,
                "num_layers": num_layers,
                "dropout_prob": dropout_prob,
                "dropout_active": dropout_active,
                "LossFn": "SoftBoundary",
            },
        ).to(self.device)

        # create the loss function

        self.c = center.to(self.device)
        self.r = radius
        self.nu = nu

        self.loss_fn = soft_boundary_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass the input through the network for a forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensors.
        """
        return self.model(x)

    def fit(self, x_train: torch.Tensor) -> None:
        """
        Perform a training step using the ``x_train`` tensor.

        Args:
            x_train: Batch of training data.
        """
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learningRate,
            weight_decay=self.weight_decay,
        )
        self.X = torch.tensor(x_train.reshape((-1, self.input_dim))).float()

        if self.train_prop != 1:
            x_train, x_val = (
                x_train[: int(self.train_prop * len(x_train))],
                x_train[int(self.train_prop * len(x_train)) :],
            )
            inputs_val = Variable(torch.from_numpy(x_val).to(self.device)).float()

        self.losses = []
        self.loss_vals = []

        for epoch in range(self.epochs):
            # Converting inputs and labels to Variable

            inputs = Variable(torch.from_numpy(x_train)).to(self.device).float()

            self.model.zero_grad()

            self.optimizer.zero_grad()

            # get output from the model, given the inputs
            outputs = self.model(inputs)

            # get loss for the predicted output
            self.loss = self.loss_fn(outputs=outputs, r=torch.Tensor([self.r]), c=self.c, nu=torch.Tensor([self.nu]))

            # self.c    = torch.mean(torch.tensor(outputs).float(), dim=0)

            # get gradients w.r.t to parameters
            self.loss.backward(retain_graph=True)
            self.losses.append(self.loss.detach().cpu().numpy())

            # update parameters
            self.optimizer.step()

            if self.train_prop != 1.0:
                with torch.no_grad():
                    # get output from the model, given the inputs
                    outputs = self.model(inputs_val)

                    # get loss for the predicted output

                    loss_val = self.loss_fn(
                        outputs=outputs, r=torch.Tensor([self.r]), c=self.c, nu=torch.Tensor([self.nu])
                    )

                    self.loss_vals.append(loss_val)

            if self.train_prop == 1:
                log(DEBUG, "epoch {}, loss {}".format(epoch, self.loss.item()))
            else:
                log(DEBUG, "epoch {:4}, train loss {:.4e}, val loss {:.4e}".format(epoch, self.loss.item(), loss_val))
