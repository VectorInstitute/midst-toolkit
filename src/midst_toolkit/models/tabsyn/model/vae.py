import math
from collections.abc import Callable
from enum import Enum

import torch
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor, nn


class Tokenizer(nn.Module):
    def __init__(self, d_numerical: int, categories: list[int] | None, d_token: int, bias: bool) -> None:
        """Initialize the tokenizer module for the VAE.

        Args:
            d_numerical: The number of numerical features.
            categories: The number of categories for each categorical feature.
                If None, the tokenizer will only use the numerical features.
            d_token: The dimension of the token.
            bias: Whether to use bias in the linear layers.
        """
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            self.categories = categories

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        """Get the number of tokens.

        Returns:
            The number of tokens.
        """
        return len(self.weight) + (0 if self.category_offsets is None else len(self.category_offsets))

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        """Forward pass of the tokenizer.

        Args:
            x_num: The numerical features.
            x_cat: The categorical features.

        Returns:
            The tokens.
        """
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )

        x = self.weight[None] * x_num[:, :, None]

        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]

        return x


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.5) -> None:
        """Initialize the MLP module for the VAE.

        Args:
            input_dim: The dimension of the input.
            hidden_dim: The dimension of the hidden layer.
            output_dim: The dimension of the output.
            dropout: The dropout rate.
        """
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the MLP.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class InitializationMethod(Enum):
    XAVIER = "xavier"
    KAIMING = "kaiming"


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d: int,
        n_heads: int,
        dropout: float,
        initialization: InitializationMethod = InitializationMethod.KAIMING,
    ) -> None:
        """Initialize the multihead attention module for the VAE.

        Args:
            d: The dimension of the input.
            n_heads: The number of heads.
            dropout: The dropout rate.
            initialization: The initialization method.
        """
        if n_heads > 1:
            assert d % n_heads == 0

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == InitializationMethod.XAVIER and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        """Reshape the input tensor.

        Args:
            x: The input tensor.

        Returns:
            The reshaped tensor.
        """
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: Callable | None = None,
        value_compression: Callable | None = None,
    ) -> Tensor:
        """Forward pass of the multihead attention.

        Args:
            x_q: The query tensor.
            x_kv: The key and value tensor.
            key_compression: The key compression function. If None, the key will not be compressed.
                If provided, the value_compression must also be provided. Optional, default is None.
            value_compression: The value compression function. If None, the value will not be compressed.
                If provided, the key_compression must also be provided. Optional, default is None.

        Returns:
            The output tensor.
        """
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)

        a = q @ k.transpose(1, 2)
        b = math.sqrt(d_head_key)
        attention = F.softmax(a / b, dim=-1)

        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_out: int,
        d_ffn_factor: int,
        attention_dropout=0.0,
        ffn_dropout=0.0,
        residual_dropout=0.0,
        prenormalization=True,
        initialization=InitializationMethod.KAIMING,
    ):
        """Initialize the transformer module for the VAE.

        Args:
            n_layers: The number of layers.
            d_token: The dimension of the token.
            n_heads: The number of heads.
            d_out: The dimension of the output.
            d_ffn_factor: The factor for the dimension of the hidden layer.
            attention_dropout: The dropout rate for the attention. Optional, default is 0.0.
            ffn_dropout: The dropout rate for the FFN. Optional, default is 0.0.
            residual_dropout: The dropout rate for the residual. Optional, default is 0.0.
            prenormalization: Whether to use pre-normalization. Optional, default is True.
            initialization: The initialization method. Optional, default is InitializationMethod.KAIMING.
        """
        super().__init__()

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    "attention": MultiheadAttention(d_token, n_heads, attention_dropout, initialization),
                    "linear0": nn.Linear(d_token, d_hidden),
                    "linear1": nn.Linear(d_hidden, d_token),
                    "norm1": make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer["norm0"] = make_normalization()

            self.layers.append(layer)

        self.activation = nn.ReLU()
        self.last_activation = nn.ReLU()
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def _start_residual(self, x: Tensor, layer: nn.ModuleDict, norm_idx: int) -> Tensor:
        """Start the residual connection.

        Args:
            x: The input tensor.
            layer: The layer.
            norm_idx: The index of the normalization layer.

        Returns:
            The residual tensor.
        """
        x_residual = x
        if self.prenormalization:
            norm_key = f"norm{norm_idx}"
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x: Tensor, x_residual: Tensor, layer: nn.ModuleDict, norm_idx: int) -> Tensor:
        """End the residual connection.

        Args:
            x: The input tensor.
            x_residual: The residual tensor.
            layer: The layer.
            norm_idx: The index of the normalization layer.

        Returns:
            The output tensor.
        """
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"norm{norm_idx}"](x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the transformer.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        for _, layer in enumerate(self.layers):
            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer["attention"](
                # for the last attention, it is enough to process only [CLS]
                x_residual,
                x_residual,
            )

            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer["linear0"](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer["linear1"](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)
        return x


class AE(nn.Module):
    def __init__(self, hid_dim: int, n_head: int) -> None:
        """Initialize the AE module for the VAE.

        Args:
            hid_dim: The dimension of the hidden layer.
            n_heads: The number of heads.
        """
        super(AE, self).__init__()

        self.hid_dim = hid_dim
        self.n_head = n_head

        self.encoder = MultiheadAttention(hid_dim, n_head)
        self.decoder = MultiheadAttention(hid_dim, n_head)

    def get_embedding(self, x: Tensor) -> Tensor:
        """Get the embedding of the input tensor.

        Args:
            x: The input tensor.

        Returns:
            The embedding tensor.
        """
        return self.encoder(x, x).detach()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the AE.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        z = self.encoder(x, x)
        h = self.decoder(z, z)

        return h


class VAE(nn.Module):
    def __init__(
        self,
        d_numerical: int,
        categories: list[int] | None,
        num_layers: int,
        hid_dim: int,
        n_head: int = 1,
        factor: int = 4,
        bias: bool = True,
    ):
        """Initialize the VAE module for the VAE.

        Args:
            d_numerical: The number of numerical features.
            categories: The number of categories for each categorical feature.
            num_layers: The number of layers.
            hid_dim: The dimension of the hidden layer.
            n_heads: The number of heads.
            factor: The factor for the dimension of the hidden layer.
            bias: Whether to use bias in the linear layers.

        Returns:
            The VAE module.
        """
        super(VAE, self).__init__()

        self.d_numerical = d_numerical
        self.categories = categories
        self.hid_dim = hid_dim
        d_token = hid_dim
        self.n_head = n_head

        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias=bias)

        self.encoder_mu = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)
        self.encoder_logvar = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)

        self.decoder = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)

    def get_embedding(self, x: Tensor) -> Tensor:
        """Get the embedding of the input tensor.

        Args:
            x: The input tensor.

        Returns:
            The embedding tensor.
        """
        return self.encoder_mu(x, x).detach()

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterize the input tensor.

        Args:
            mu: The mean tensor.
            logvar: The log variance tensor.

        Returns:
            The reparameterized tensor.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        """Forward pass of the VAE.

        Args:
            x_num: The numerical features.
            x_cat: The categorical features.

        Returns:
            The output tensor.
        """
        x = self.Tokenizer(x_num, x_cat)

        mu_z = self.encoder_mu(x)
        std_z = self.encoder_logvar(x)

        z = self.reparameterize(mu_z, std_z)

        h = self.decoder(z[:, 1:])

        return h, mu_z, std_z


class Reconstructor(nn.Module):
    def __init__(self, d_numerical: int, categories: int, d_token: int) -> None:
        """Initialize the reconstructor module for the VAE.

        Args:
            d_numerical: The number of numerical features.
            categories: The number of categories for each categorical feature.
            d_token: The dimension of the token.
        """
        super(Reconstructor, self).__init__()

        self.d_numerical = d_numerical
        self.categories = categories
        self.d_token = d_token

        self.weight = nn.Parameter(Tensor(d_numerical, d_token))
        nn.init.xavier_uniform_(self.weight, gain=1 / math.sqrt(2))
        self.cat_recons = nn.ModuleList()

        for d in categories:
            recon = nn.Linear(d_token, d)
            nn.init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))
            self.cat_recons.append(recon)

    def forward(self, h: Tensor) -> Tensor:
        """Forward pass of the reconstructor.

        Args:
            h: The input tensor.

        Returns:
            The output tensor.
        """
        h_num = h[:, : self.d_numerical]
        h_cat = h[:, self.d_numerical :]

        recon_x_num = torch.mul(h_num, self.weight.unsqueeze(0)).sum(-1)
        recon_x_cat = []

        for i, recon in enumerate(self.cat_recons):
            recon_x_cat.append(recon(h_cat[:, i]))

        return recon_x_num, recon_x_cat


class Model_VAE(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_numerical: int,
        categories: list[int] | None,
        d_token: int,
        n_head: int = 1,
        factor: int = 4,
        bias: bool = True,
    ) -> None:
        """Initialize the VAE Model.

        Args:
            num_layers: The number of layers.
            d_numerical: The number of numerical features.
            categories: The number of categories for each categorical feature.
            d_token: The dimension of the token.
            n_head: The number of heads.
            factor: The factor for the dimension of the hidden layer.
            bias: Whether to use bias in the linear layers.
        """
        super(Model_VAE, self).__init__()

        self.VAE = VAE(
            d_numerical,
            categories,
            num_layers,
            d_token,
            n_head=n_head,
            factor=factor,
            bias=bias,
        )
        self.Reconstructor = Reconstructor(d_numerical, categories, d_token)

    def get_embedding(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        """Get the embedding of the input tensor.

        Args:
            x_num: The numerical features.
            x_cat: The categorical features.

        Returns:
            The embedding tensor.
        """
        x = self.Tokenizer(x_num, x_cat)
        return self.VAE.get_embedding(x)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        """Forward pass of the VAE.

        Args:
            x_num: The numerical features.
            x_cat: The categorical features.

        Returns:
            The output tensor.
        """
        h, mu_z, std_z = self.VAE(x_num, x_cat)

        # recon_x_num, recon_x_cat = self.Reconstructor(h[:, 1:])
        recon_x_num, recon_x_cat = self.Reconstructor(h)

        return recon_x_num, recon_x_cat, mu_z, std_z


class Encoder_model(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_numerical: int,
        categories: list[int] | None,
        d_token: int,
        n_head: int = 1,
        factor: int = 4,
        bias: bool = True,
    ) -> None:
        """Initialize the Encoder model.

        Args:
            num_layers: The number of layers.
            d_numerical: The number of numerical features.
            categories: The number of categories for each categorical feature.
            d_token: The dimension of the token.
            n_head: The number of heads.
            factor: The factor for the dimension of the hidden layer.
            bias: Whether to use bias in the linear layers.
        """
        super().__init__()
        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias)
        self.VAE_Encoder = Transformer(num_layers, d_token, n_head, d_token, factor)

    def load_weights(self, pretrained_VAE: nn.Module) -> None:
        """Load the weights of the encoder model.

        Args:
            pretrained_VAE: The pretrained VAE model.
        """
        self.Tokenizer.load_state_dict(pretrained_VAE.VAE.Tokenizer.state_dict())
        self.VAE_Encoder.load_state_dict(pretrained_VAE.VAE.encoder_mu.state_dict())

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        """Forward pass of the encoder model.

        Args:
            x_num: The numerical features.
            x_cat: The categorical features.

        Returns:
            The output tensor.
        """
        x = self.Tokenizer(x_num, x_cat)
        z = self.VAE_Encoder(x)

        return z


class Decoder_model(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_numerical: int,
        categories: list[int],
        d_token: int,
        n_head: int = 1,
        factor: int = 4,
        bias: bool = True,
    ) -> None:
        """Initialize the Decoder model.

        Args:
            num_layers: The number of layers.
            d_numerical: The number of numerical features.
            categories: The number of categories for each categorical feature.
            d_token: The dimension of the token.
            n_head: The number of heads.
            factor: The factor for the dimension of the hidden layer.
            bias: Whether to use bias in the linear layers.
        """
        super(Decoder_model, self).__init__()
        self.VAE_Decoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.Detokenizer = Reconstructor(d_numerical, categories, d_token)

    def load_weights(self, pretrained_VAE: nn.Module) -> None:
        """Load the weights of the decoder model.

        Args:
            pretrained_VAE: The pretrained VAE model.
        """
        self.VAE_Decoder.load_state_dict(pretrained_VAE.VAE.decoder.state_dict())
        self.Detokenizer.load_state_dict(pretrained_VAE.Reconstructor.state_dict())

    def forward(self, z: Tensor) -> Tensor:
        """Forward pass of the decoder model.

        Args:
            z: The input tensor.

        Returns:
            The output tensor.
        """
        h = self.VAE_Decoder(z)
        x_hat_num, x_hat_cat = self.Detokenizer(h)

        return x_hat_num, x_hat_cat
