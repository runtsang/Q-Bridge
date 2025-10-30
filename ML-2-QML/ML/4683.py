"""Hybrid classical estimator combining auto‑encoding, self‑attention and regression.

The network first projects inputs into a latent space with a
lightweight auto‑encoder, then applies a learnable self‑attention
module to capture contextual dependencies.  The attended representation
is finally mapped to a scalar prediction.  The design mirrors the
quantum reference while remaining fully classical and scalable."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: torch.Tensor | list[float] | np.ndarray) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)


@dataclass
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""

    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """A lightweight multilayer perceptron auto‑encoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Factory that returns a configured auto‑encoder."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)


class ClassicalSelfAttention(nn.Module):
    """Learnable self‑attention layer."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable projection matrices
        self.rotation_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Shape ``(batch, embed_dim)``

        Returns
        -------
        torch.Tensor
            Attention‑weighted representation of shape ``(batch, embed_dim)``.
        """
        query = inputs @ self.rotation_params
        key = inputs @ self.entangle_params
        scores = torch.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs


class EstimatorQNNGenNet(nn.Module):
    """Hybrid estimator that uses an auto‑encoder, self‑attention and a linear head."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        attention_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.autoencoder = Autoencoder(
            input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.attention_dim = attention_dim or latent_dim
        self.attention = ClassicalSelfAttention(self.attention_dim)
        self.regressor = nn.Linear(self.attention_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that compresses, attends, and regresses.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape ``(batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Scalar predictions of shape ``(batch,)``.
        """
        latents = self.autoencoder.encode(x)
        attended = self.attention(latents)
        out = self.regressor(attended)
        return out.squeeze(-1)


def EstimatorQNNGen(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    attention_dim: int | None = None,
) -> EstimatorQNNGenNet:
    """Factory that returns a fully configured hybrid estimator."""
    return EstimatorQNNGenNet(
        input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        attention_dim=attention_dim,
    )


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Simple reconstruction training loop returning the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


def train_regressor(
    model: EstimatorQNNGenNet,
    data: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop for the hybrid estimator."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data), _as_tensor(targets))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
    "ClassicalSelfAttention",
    "EstimatorQNNGen",
    "EstimatorQNNGenNet",
    "train_regressor",
]
