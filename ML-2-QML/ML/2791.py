"""Hybrid classical autoencoder with an optional quantum layer.

The module defines :class:`AutoencoderHybridNet`, a PyTorch ``nn.Module`` that
builds a standard encoder/decoder stack and, if supplied, runs a quantum
sub‑network on the latent representation.  The quantum sub‑network is
expected to be a callable that accepts a ``torch.Tensor`` and returns a
``torch.Tensor`` of the same shape.  The helper :func:`AutoencoderHybrid`
creates a configured instance, and :func:`train_autoencoder_hybrid`
provides a simple reconstruction training loop.

Typical usage:

>>> from Autoencoder__gen046 import AutoencoderHybrid, train_autoencoder_hybrid
>>> model = AutoencoderHybrid(784, latent_dim=32)
>>> history = train_autoencoder_hybrid(model, train_data)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclass
class AutoencoderHybridConfig:
    """Configuration for :class:`AutoencoderHybridNet`."""

    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    quantum_layer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


class AutoencoderHybridNet(nn.Module):
    """A lightweight multilayer perceptron autoencoder with an optional quantum layer."""

    def __init__(self, config: AutoencoderHybridConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder
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

        # Optional quantum layer
        self.quantum_layer = config.quantum_layer

        # Decoder
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        if self.quantum_layer is not None:
            z = self.quantum_layer(z)
        return self.decode(z)


def AutoencoderHybrid(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    quantum_layer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> AutoencoderHybridNet:
    """Factory returning a configured :class:`AutoencoderHybridNet`."""
    config = AutoencoderHybridConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        quantum_layer=quantum_layer,
    )
    return AutoencoderHybridNet(config)


def train_autoencoder_hybrid(
    model: AutoencoderHybridNet,
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


__all__ = [
    "AutoencoderHybrid",
    "AutoencoderHybridNet",
    "AutoencoderHybridConfig",
    "train_autoencoder_hybrid",
]
