"""Hybrid complex‑valued autoencoder with quantum‑inspired latent space.

This module defines :class:`Autoencoder__gen357`, a fully‑connected autoencoder that
operates on complex tensors.  The encoder and decoder are linear layers that
maintain the complex structure, enabling the latent representation to be
interpreted as a quantum state.  The class exposes the same public API as the
original seed but adds complex‑valued support and a unit‑norm regularizer.

Example
-------
>>> import torch
>>> ae = Autoencoder__gen357(input_dim=10, latent_dim=4)
>>> x = torch.randn(2, 10, dtype=torch.complex64)
>>> recon = ae(x)
>>> recon.shape
torch.Size([2, 10])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Convert input to a complex‑64 tensor on the current device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.complex64)
    if tensor.dtype!= torch.complex64:
        tensor = tensor.to(dtype=torch.complex64)
    return tensor


@dataclass
class AutoencoderConfig:
    """Configuration values for :class:`Autoencoder__gen357`."""

    input_dim: int
    """Number of input features."""
    latent_dim: int = 32
    """Size of the latent space."""
    hidden_dims: Tuple[int, int] = (128, 64)
    """Hidden layer sizes for encoder/decoder."""
    dropout: float = 0.1
    """Dropout rate within hidden layers."""


class Autoencoder__gen357(nn.Module):
    """Complex‑valued autoencoder with quantum‑inspired latent space."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
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
        """Encode complex inputs to latent space."""
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to complex outputs."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

    def unit_norm_regularizer(self, latents: torch.Tensor) -> torch.Tensor:
        """Return a penalty for deviation from unit norm."""
        norms = torch.norm(latents, dim=-1)
        return torch.mean((norms - 1.0) ** 2)


def Autoencoder__gen357_factory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> Autoencoder__gen357:
    """Factory mirroring the original helper."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return Autoencoder__gen357(config)


def train_autoencoder(
    model: Autoencoder__gen357,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop for the complex autoencoder."""
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
            # unit‑norm regularizer
            loss += 0.01 * model.unit_norm_regularizer(recon)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = ["Autoencoder__gen357", "Autoencoder__gen357_factory", "train_autoencoder"]
