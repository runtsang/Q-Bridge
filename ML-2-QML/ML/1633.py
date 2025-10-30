"""
Hybrid autoencoder with optional quantum refinement layer.

This module provides a purely classical implementation that can
be extended with a quantum refinement layer.  The classical part
is identical to the original seed but accepts an optional callable
`quantum_layer` that is applied to the latent representation.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class AutoencoderConfig:
    """Configuration for :class:`Autoencoder__gen204`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


# --------------------------------------------------------------------------- #
# Core model
# --------------------------------------------------------------------------- #
class Autoencoder__gen204(nn.Module):
    """Hybrid autoencoder with an optional quantum refinement step."""
    def __init__(
        self,
        config: AutoencoderConfig,
        quantum_layer: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.quantum_layer = quantum_layer

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

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def refine_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.quantum_layer is not None:
            return self.quantum_layer(latent)
        return latent

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encode(inputs)
        refined = self.refine_latent(latent)
        return self.decode(refined)


# --------------------------------------------------------------------------- #
# Training helper
# --------------------------------------------------------------------------- #
def train_autoencoder(
    model: Autoencoder__gen204,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Train a hybrid autoencoder and return the loss history."""
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
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

    return history


__all__ = ["Autoencoder__gen204", "AutoencoderConfig", "train_autoencoder"]
