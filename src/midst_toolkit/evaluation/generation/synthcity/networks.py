# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import torch
from torch import nn


ACTIVATION_DICT = {
    "ReLU": torch.nn.ReLU(),
    "Hardtanh": torch.nn.Hardtanh(),
    "ReLU6": torch.nn.ReLU6(),
    "Sigmoid": torch.nn.Sigmoid(),
    "Tanh": torch.nn.Tanh(),
    "ELU": torch.nn.ELU(),
    "CELU": torch.nn.CELU(),
    "SELU": torch.nn.SELU(),
    "GLU": torch.nn.GLU(),
    "LeakyReLU": torch.nn.LeakyReLU(),
    "LogSigmoid": torch.nn.LogSigmoid(),
    "Softplus": torch.nn.Softplus(),
}


def build_network(network_name: str, params: dict) -> nn.Module:
    """
    Placeholder for now. Would be a factory type method if there where more than one option.

    Args:
        network_name: Name of the network to be generated.
        params: Parameters/configuration used to create the network in a custom way.

    Returns:
        A torch module to be trained.
    """
    if network_name == "feedforward":
        return feedforward_network(params)
    raise ValueError("Network name not recognized.")


def feedforward_network(params: dict) -> nn.Module:
    """
    Architecture for a Feedforward Neural Network.

    Args:
        params: Has keys ["input_dim", "rep_dim", "num_hidden", "activation", "num_layers", "dropout_prob",
            "dropout_active", "LossFn". These determine the architecture structure

    Returns:
        The constructed network.
    """
    modules: list[nn.Module] = []

    if params["dropout_active"]:
        modules.append(torch.nn.Dropout(p=params["dropout_prob"]))

    # Input layer

    modules.append(torch.nn.Linear(params["input_dim"], params["num_hidden"], bias=False))
    modules.append(ACTIVATION_DICT[params["activation"]])

    # Intermediate layers

    for _ in range(params["num_layers"] - 1):
        if params["dropout_active"]:
            modules.append(torch.nn.Dropout(p=params["dropout_prob"]))

        modules.append(torch.nn.Linear(params["num_hidden"], params["num_hidden"], bias=False))
        modules.append(ACTIVATION_DICT[params["activation"]])

    # Output layer

    modules.append(torch.nn.Linear(params["num_hidden"], params["rep_dim"], bias=False))

    return nn.Sequential(*modules)
