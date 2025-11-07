from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from logging import INFO
from typing import Any, Self

import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional

from midst_toolkit.common.enumerations import DomainDataType, TaskType
from midst_toolkit.common.logger import log
from midst_toolkit.models.clavaddpm.enumerations import IsTargetConditioned, ModuleType


@dataclass
class DiffusionParameters:
    """Parameters for the diffusion model."""

    layers_dimensions: list[int]
    dropout: float
    input_dimension: int = 0
    output_dimension: int = 0
    embedding_dimension: int = 0
    n_blocks: int = 0
    block_dimension: int = 0
    hidden_dimension: int = 0
    dropout_first: float = 0
    dropout_second: float = 0


@dataclass
class ModelParameters:
    """Parameters for the ClavaDDPM model."""

    diffusion_parameters: DiffusionParameters
    input_dimension: int = 0
    num_classes: int = 0
    is_target_conditioned: IsTargetConditioned = IsTargetConditioned.NONE


class Classifier(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        timestep_dimension: int,
        hidden_sizes: list[int],
        dropout_prob: float = 0.5,
        num_heads: int = 2,
        num_layers: int = 1,
    ):
        """
        Initialize the classifier model.

        Args:
            input_dimension: The input dimension size.
            output_dimension: The output dimension size.
            timestep_dimension: The dimension size of the timestep.
            hidden_sizes: The list of sizes for the hidden layers.
            dropout_prob: The dropout probability. Optional, default is 0.5.
            num_heads: The number of heads for the transformer layer. Optional, default is 2.
            num_layers: The number of layers for the transformer layer. Optional, default is 1.
        """
        super(Classifier, self).__init__()

        self.timestep_dimension = timestep_dimension
        self.proj = nn.Linear(input_dimension, timestep_dimension)

        self.transformer_layer = nn.Transformer(
            d_model=timestep_dimension,
            nhead=num_heads,
            num_encoder_layers=num_layers,
        )

        self.timestep_embedding = nn.Sequential(
            nn.Linear(timestep_dimension, timestep_dimension),
            nn.SiLU(),
            nn.Linear(timestep_dimension, timestep_dimension),
        )

        # Create a list to hold the layers
        layers: list[nn.Module] = []

        # Add input layer
        layers.append(nn.Linear(timestep_dimension, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))  # Batch Normalization
        layers.append(nn.Dropout(p=dropout_prob))

        # Add hidden layers with batch normalization and different activation
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.LeakyReLU())  # Different activation
            layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))  # Batch Normalization
            layers.append(nn.Dropout(p=dropout_prob))

        # Add output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_dimension))

        # Create a Sequential model from the list of layers
        self.model = nn.Sequential(*layers)

    def forward(self, input_tensor: Tensor, timesteps: Tensor) -> Tensor:
        """
        Forward pass of the classifier model.

        Args:
            input_tensor: The input tensor.
            timesteps: The timesteps tensor.

        Returns:
            The output tensor.
        """
        embeddings = self.timestep_embedding(timestep_embedding(timesteps, self.timestep_dimension))
        output_tensor = self.proj(input_tensor) + embeddings
        return self.model(output_tensor)


def get_table_info(df: pd.DataFrame, table_domain: dict[str, Any], target_column_name: str) -> dict[str, Any]:
    """
    Get the dictionary of table information.

    Args:
        df: The dataframe containing the data.
        table_domain: The table's domain dictionary containing metadata about the data columns.
        target_column_name: The name of the target column.

    Returns:
        The table information in the following format:
        {
            "cat_cols": list[str],
            "num_cols": list[str],
            "y_col": str,
            "n_classes": int,
            "task_type": str,
        }
    """
    categorical_cols = []
    numerical_cols = []
    for column in df.columns:
        if column in table_domain and column != target_column_name:
            if table_domain[column]["type"] == DomainDataType.DISCRETE.value:
                categorical_cols.append(column)
            else:
                numerical_cols.append(column)

    table_info: dict[str, Any] = {}
    table_info["cat_cols"] = categorical_cols
    table_info["num_cols"] = numerical_cols
    table_info["y_col"] = target_column_name
    table_info["n_classes"] = 0
    table_info["task_type"] = TaskType.MULTICLASS_CLASSIFICATION.value

    return table_info


def timestep_embedding(timesteps: Tensor, output_dimension: int, max_period: int = 10000) -> Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
        output_dimension: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.

    Returns:
        An [N x output_dimension] Tensor of positional embeddings.
    """
    half = output_dimension // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
    freqs = freqs.to(device=timesteps.device)

    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if output_dimension % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


class MLP(nn.Module):
    """The MLP model used in [gorishniy2021revisiting].

    The following scheme describes the architecture:

    .. code-block:: text

          MLP: (in) -> Block -> ... -> Block -> Linear -> (out)
        Block: (in) -> Linear -> Activation -> Dropout -> (out)

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            module = MLP.make_baseline(x.shape[1], [3, 5], 0.1, 1)
            assert module(x).shape == (len(x), 1)

    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov,
        Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `MLP`."""

        def __init__(
            self,
            *,
            input_dimension: int,
            output_dimension: int,
            bias: bool,
            activation: ModuleType,
            dropout: float,
        ) -> None:
            """
            Initialize the MLP block.

            Args:
                input_dimension: The input dimension size.
                output_dimension: The output dimension size.
                bias: Whether to use bias.
                activation: The activation function.
                dropout: The dropout probability.
            """
            super().__init__()
            self.linear = nn.Linear(input_dimension, output_dimension, bias)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, input_tensor: Tensor) -> Tensor:
            """
            Forward pass of the MLP block.

            Args:
                input_tensor: The input tensor.

            Returns:
                The output tensor.
            """
            return self.dropout(self.activation(self.linear(input_tensor)))

    def __init__(
        self,
        *,
        input_dimension: int,
        layers_dimensions: list[int],
        dropouts: float | list[float],
        activation: ModuleType,
        output_dimension: int,
    ):
        """
        Initialize the MLP model.

        Note:
            `make_baseline` is the recommended constructor.

        Args:
            input_dimension: The input dimension size.
            layers_dimensions: The list of sizes for the hidden layers.
            dropouts: Can be either a single value for the dropout rate or a list of dropout rates.
            activation: The activation function.
            output_dimension: The output dimension size.
        """
        super().__init__()

        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(layers_dimensions)

        assert len(layers_dimensions) == len(dropouts)
        assert activation not in [ModuleType.REGLU, ModuleType.GEGLU]

        self.blocks = nn.ModuleList(
            [
                MLP.Block(
                    input_dimension=layers_dimensions[i - 1] if i else input_dimension,
                    output_dimension=d,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(layers_dimensions, dropouts))
            ]
        )
        self.head = nn.Linear(layers_dimensions[-1] if layers_dimensions else input_dimension, output_dimension)

    @classmethod
    def make_baseline(
        cls,
        input_dimension: int,
        layers_dimensions: list[int],
        dropout: float,
        output_dimension: int,
    ) -> Self:
        """Create a "baseline" `MLP`.

        This variation of MLP was used in [gorishniy2021revisiting]. Features:

        * all linear layers except for the first one and the last one are of the same dimension
        * the dropout rate is the same for all dropout layers

        Args:
            input_dimension: the input size
            layers_dimensions: the dimensions of the linear layers. If there are more than two
                layers, then all of them except for the first and the last ones must
                have the same dimension.
            dropout: the dropout rate for all hidden layers
            output_dimension: the output size
        Returns:
            MLP

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov,
            Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        assert isinstance(dropout, float)
        if len(layers_dimensions) > 2:
            assert len(set(layers_dimensions[1:-1])) == 1, (
                "if d_layers contains more than two elements, then"
                " all elements except for the first and the last ones must be equal."
            )
        return cls(
            input_dimension=input_dimension,
            layers_dimensions=layers_dimensions,
            dropouts=dropout,
            activation=ModuleType.RELU,
            output_dimension=output_dimension,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Forward pass of the MLP model.

        Args:
            input_tensor: The input tensor.

        Returns:
            The output tensor.
        """
        input_tensor = input_tensor.float()

        for block in self.blocks:
            input_tensor = block(input_tensor)

        return self.head(input_tensor)


class ResNet(nn.Module):
    """
    The ResNet model used in [gorishniy2021revisiting].

    The following scheme describes the architecture:
    .. code-block:: text
        ResNet: (in) -> Linear -> Block -> ... -> Block -> Head -> (out)
                 |-> Norm -> Linear -> Activation -> Dropout -> Linear -> Dropout ->|
                 |                                                                  |
         Block: (in) ------------------------------------------------------------> Add -> (out)
          Head: (in) -> Norm -> Activation -> Linear -> (out)

    Examples:
        .. testcode::
            x = torch.randn(4, 2)
            module = ResNet.make_baseline(
                input_dimension=x.shape[1],
                n_blocks=2,
                block_dimension=3,
                hidden_dimension=4,
                dropout_first=0.25,
                dropout_second=0.0,
                output_dimension=1
            )
            assert module(x).shape == (len(x), 1)

    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov,
        Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `ResNet`."""

        def __init__(
            self,
            *,
            block_dimension: int,
            hidden_dimension: int,
            bias_first: bool,
            bias_second: bool,
            dropout_first: float,
            dropout_second: float,
            normalization: ModuleType,
            activation: ModuleType,
            skip_connection: bool,
        ):
            """
            Initialize the ResNet block.

            Args:
                block_dimension: The input dimension size.
                hidden_dimension: The output dimension size.
                bias_first: Whether to use bias for the first linear layer.
                bias_second: Whether to use bias for the second linear layer.
                dropout_first: The dropout probability for the first dropout layer.
                dropout_second: The dropout probability for the second dropout layer.
                normalization: The normalization function.
                activation: The activation function.
                skip_connection: Whether to use skip connection.
            """
            super().__init__()
            self.normalization = _make_nn_module(normalization, block_dimension)
            self.linear_first = nn.Linear(block_dimension, hidden_dimension, bias_first)
            self.activation = _make_nn_module(activation)
            self.dropout_first = nn.Dropout(dropout_first)
            self.linear_second = nn.Linear(hidden_dimension, block_dimension, bias_second)
            self.dropout_second = nn.Dropout(dropout_second)
            self.skip_connection = skip_connection

        def forward(self, input_tensor: Tensor) -> Tensor:
            """
            Forward pass of the ResNet block.

            Args:
                input_tensor: The input tensor.

            Returns:
                The output tensor.
            """
            x_input = input_tensor
            input_tensor = self.normalization(input_tensor)
            input_tensor = self.linear_first(input_tensor)
            input_tensor = self.activation(input_tensor)
            input_tensor = self.dropout_first(input_tensor)
            input_tensor = self.linear_second(input_tensor)
            input_tensor = self.dropout_second(input_tensor)

            if self.skip_connection:
                input_tensor = x_input + input_tensor

            return input_tensor

    class Head(nn.Module):
        """The final module of `ResNet`."""

        def __init__(
            self,
            *,
            input_dimension: int,
            output_dimension: int,
            bias: bool,
            normalization: ModuleType,
            activation: ModuleType,
        ):
            """
            Initialize the ResNet head.

            Args:
                input_dimension: The input dimension size.
                output_dimension: The output dimension size.
                bias: Whether to use bias.
                normalization: The normalization function.
                activation: The activation function.
            """
            super().__init__()
            self.normalization = _make_nn_module(normalization, input_dimension)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(input_dimension, output_dimension, bias)

        def forward(self, input_tensor: Tensor) -> Tensor:
            """
            Forward pass of the ResNet head.

            Args:
                input_tensor: The input tensor.

            Returns:
                The output tensor.
            """
            if self.normalization is not None:
                input_tensor = self.normalization(input_tensor)

            input_tensor = self.activation(input_tensor)
            return self.linear(input_tensor)

    def __init__(
        self,
        *,
        input_dimension: int,
        n_blocks: int,
        block_dimension: int,
        hidden_dimension: int,
        dropout_first: float,
        dropout_second: float,
        normalization: ModuleType,
        activation: ModuleType,
        output_dimension: int,
    ):
        """
        Initialize the ResNet model.

        Note:
            `make_baseline` is the recommended constructor.

        Args:
            input_dimension: The input dimension size.
            n_blocks: The number of blocks.
            block_dimension: The input dimension size.
            hidden_dimension: The output dimension size.
            dropout_first: The dropout probability for the first dropout layer.
            dropout_second: The dropout probability for the second dropout layer.
            normalization: The normalization function.
            activation: The activation function.
            output_dimension: The output dimension size.
        """
        super().__init__()

        self.first_layer = nn.Linear(input_dimension, block_dimension)

        if block_dimension is None:
            block_dimension = input_dimension

        self.blocks = nn.Sequential(
            *[
                ResNet.Block(
                    block_dimension=block_dimension,
                    hidden_dimension=hidden_dimension,
                    bias_first=True,
                    bias_second=True,
                    dropout_first=dropout_first,
                    dropout_second=dropout_second,
                    normalization=normalization,
                    activation=activation,
                    skip_connection=True,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = ResNet.Head(
            input_dimension=block_dimension,
            output_dimension=output_dimension,
            bias=True,
            normalization=normalization,
            activation=activation,
        )

    @classmethod
    def make_baseline(
        cls,
        *,
        input_dimension: int,
        n_blocks: int,
        block_dimension: int,
        hidden_dimension: int,
        dropout_first: float,
        dropout_second: float,
        output_dimension: int,
    ) -> Self:
        """
        Create a "baseline" `ResNet`. This variation of ResNet was used in [gorishniy2021revisiting].

        Args:
            input_dimension: the input size
            n_blocks: the number of Blocks
            block_dimension: the input size (or, equivalently, the output size) of each Block
            hidden_dimension: the output size of the first linear layer in each Block
            dropout_first: the dropout rate of the first dropout layer in each Block.
            dropout_second: the dropout rate of the second dropout layer in each Block.
            output_dimension: Output dimension.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov,
            Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        return cls(
            input_dimension=input_dimension,
            n_blocks=n_blocks,
            block_dimension=block_dimension,
            hidden_dimension=hidden_dimension,
            dropout_first=dropout_first,
            dropout_second=dropout_second,
            normalization=ModuleType.BATCH_NORM_1D,
            activation=ModuleType.RELU,
            output_dimension=output_dimension,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Forward pass of the ResNet model.

        Args:
            input_tensor: The input tensor.

        Returns:
            The output tensor.
        """
        input_tensor = input_tensor.float()
        input_tensor = self.first_layer(input_tensor)
        input_tensor = self.blocks(input_tensor)
        return self.head(input_tensor)


#### For diffusion


class MLPDiffusion(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        num_classes: int,
        is_target_conditioned: IsTargetConditioned,
        diffusion_parameters: DiffusionParameters,
        timestep_dimension: int = 128,
    ):
        """
        Initialize the MLP diffusion model.

        Args:
            input_dimension: The input dimension size.
            num_classes: The number of classes.
            is_target_conditioned: The condition on the model target.
            diffusion_parameters: The parameters for the MLP.
            timestep_dimension: The dimension size of the timestep.
        """
        super().__init__()
        self.timestep_dimension = timestep_dimension
        self.num_classes = num_classes
        self.is_target_conditioned = is_target_conditioned

        self.diffusion_parameters = diffusion_parameters
        self.diffusion_parameters.input_dimension = timestep_dimension
        self.diffusion_parameters.output_dimension = input_dimension

        self.mlp = MLP.make_baseline(
            input_dimension=self.diffusion_parameters.input_dimension,
            layers_dimensions=self.diffusion_parameters.layers_dimensions,
            dropout=self.diffusion_parameters.dropout,
            output_dimension=self.diffusion_parameters.output_dimension,
        )

        self.label_embedding: nn.Embedding | nn.Linear
        if self.num_classes > 0 and is_target_conditioned == IsTargetConditioned.EMBEDDING:
            self.label_embedding = nn.Embedding(self.num_classes, timestep_dimension)
        elif self.num_classes == 0 and is_target_conditioned == IsTargetConditioned.EMBEDDING:
            self.label_embedding = nn.Linear(1, timestep_dimension)

        self.proj = nn.Linear(input_dimension, timestep_dimension)
        self.timestep_embedding = nn.Sequential(
            nn.Linear(timestep_dimension, timestep_dimension),
            nn.SiLU(),
            nn.Linear(timestep_dimension, timestep_dimension),
        )

    def forward(self, input_tensor: Tensor, timesteps: Tensor, y: Tensor | None = None) -> Tensor:
        """
        Forward pass of the MLP diffusion model.

        Args:
            input_tensor: The input tensor.
            timesteps: The timesteps tensor.
            y: The y tensor. Optional, default is None.

        Returns:
            The output tensor.
        """
        embeddings = self.timestep_embedding(timestep_embedding(timesteps, self.timestep_dimension))

        if self.is_target_conditioned == IsTargetConditioned.EMBEDDING and y is not None:
            y = y.squeeze() if self.num_classes > 0 else y.resize_(y.size(0), 1).float()
            embeddings += functional.silu(self.label_embedding(y))

        input_tensor = self.proj(input_tensor) + embeddings
        return self.mlp(input_tensor)


class ResNetDiffusion(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        num_classes: int,
        diffusion_parameters: DiffusionParameters,
        timestep_dimension: int = 256,
        is_target_conditioned: IsTargetConditioned | None = None,
    ):
        """
        Initialize the ResNet diffusion model.

        Args:
            input_dimension: The input dimension size.
            num_classes: The number of classes.
            diffusion_parameters: The parameters for the ResNet.
            timestep_dimension: The dimension size of the timestep.
            is_target_conditioned: The condition on the model target. Optional, default is None.
        """
        super().__init__()
        self.timestep_dimension = timestep_dimension
        self.num_classes = num_classes
        self.is_target_conditioned = is_target_conditioned

        self.diffusion_parameters = diffusion_parameters
        self.diffusion_parameters.input_dimension = input_dimension
        self.diffusion_parameters.output_dimension = input_dimension
        self.diffusion_parameters.embedding_dimension = timestep_dimension

        self.resnet = ResNet.make_baseline(
            input_dimension=self.diffusion_parameters.input_dimension,
            n_blocks=self.diffusion_parameters.n_blocks,
            block_dimension=self.diffusion_parameters.block_dimension,
            hidden_dimension=self.diffusion_parameters.hidden_dimension,
            dropout_first=self.diffusion_parameters.dropout_first,
            dropout_second=self.diffusion_parameters.dropout_second,
            output_dimension=self.diffusion_parameters.output_dimension,
        )

        self.label_embedding: nn.Embedding | nn.Linear
        if self.num_classes > 0 and is_target_conditioned == IsTargetConditioned.EMBEDDING:
            self.label_embedding = nn.Embedding(self.num_classes, timestep_dimension)
        elif self.num_classes == 0 and is_target_conditioned == IsTargetConditioned.EMBEDDING:
            self.label_embedding = nn.Linear(1, timestep_dimension)

        self.timestep_embedding = nn.Sequential(
            nn.Linear(timestep_dimension, timestep_dimension),
            nn.SiLU(),
            nn.Linear(timestep_dimension, timestep_dimension),
        )

    def forward(self, input_tensor: Tensor, timesteps: Tensor, y: Tensor | None = None) -> Tensor:
        """
        Forward pass of the ResNet diffusion model.

        Args:
            input_tensor: The input tensor.
            timesteps: The timesteps tensor.
            y: The y tensor. Optional, default is None.

        Returns:
            The output tensor.
        """
        embeddings = self.timestep_embedding(timestep_embedding(timesteps, self.timestep_dimension))

        if y is not None and self.num_classes > 0:
            embeddings += self.label_embedding(y.squeeze())

        return self.resnet(input_tensor, embeddings)


class ReGLU(nn.Module):
    """
    The ReGLU activation function from [shazeer2020glu].

    Examples:
        module = ReGLU()
        x = torch.randn(3, 4)
        assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020

    Args:
        input_tensor: The input tensor.

    Returns:
        The output tensor.
    """

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Forward pass of the ReGLU activation function from [1].

        References:
            [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020.

        Args:
            input_tensor: The input tensor.

        Returns:
            The output tensor.
        """
        assert input_tensor.shape[-1] % 2 == 0
        a, b = input_tensor.chunk(2, dim=-1)
        return a * functional.relu(b)


class GEGLU(nn.Module):
    """
    The GEGLU activation function from [shazeer2020glu].

    Examples:
            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Forward pass of the GEGLU activation function from [1].

        References:
            [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020.

        Args:
            input_tensor: The input tensor.

        Returns:
            The output tensor.
        """
        assert input_tensor.shape[-1] % 2 == 0
        a, b = input_tensor.chunk(2, dim=-1)
        return a * functional.gelu(b)


def _make_nn_module(module_type: ModuleType | Callable[..., nn.Module], *args: Any) -> nn.Module:
    """
    Make a neural network module.

    Args:
        module_type: The type of the module. Can be one of the predefined modules types in
            ModuleType or a callable function with a custom implementation of the module.
        args: The arguments for the module.

    Returns:
        The neural network module.
    """
    if not isinstance(module_type, ModuleType):
        return module_type(*args)

    if module_type == ModuleType.REGLU:
        return ReGLU()
    if module_type == ModuleType.GEGLU:
        return GEGLU()

    return getattr(nn, module_type.value)(*args)


class ModelType(Enum):
    """Possible model types for the ClavaDDPM model."""

    MLP = "mlp"
    RESNET = "resnet"

    def get_model(self, model_parameters: ModelParameters) -> nn.Module:
        """
        Get the model.

        Args:
            model_parameters: The parameters of the model.

        Returns:
            The model.
        """
        log(INFO, f"Getting model: {self.value}")
        if self == ModelType.MLP:
            return MLPDiffusion(
                input_dimension=model_parameters.input_dimension,
                num_classes=model_parameters.num_classes,
                is_target_conditioned=model_parameters.is_target_conditioned,
                diffusion_parameters=model_parameters.diffusion_parameters,
            )
        if self == ModelType.RESNET:
            return ResNetDiffusion(
                input_dimension=model_parameters.input_dimension,
                num_classes=model_parameters.num_classes,
                diffusion_parameters=model_parameters.diffusion_parameters,
            )

        raise ValueError(f"Unsupported model type: {self.value}")
