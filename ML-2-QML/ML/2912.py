"""Hybrid autoencoder – classical PyTorch implementation.

This module implements a purely classical network that mirrors the
interface of the quantum version.  It can be used as a stand‑alone
autoencoder or as a drop‑in replacement for the quantum encoder in a
hybrid workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

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
class HybridAutoencoderConfig:
    """Configuration for :class:`HybridAutoencoderNet`."""
    input_shape: Tuple[int, int, int]  # (channels, height, width)
    latent_dim: int = 32
    conv_hidden_channels: Tuple[int, int] = (8, 16)
    conv_kernel_size: int = 3
    dropout: float = 0.1


class HybridAutoencoderNet(nn.Module):
    """Convolutional encoder → linear latent → fully‑connected decoder."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.input_shape = config.input_shape
        c, h, w = config.input_shape

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(c, config.conv_hidden_channels[0],
                      kernel_size=config.conv_kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(config.conv_hidden_channels[0], config.conv_hidden_channels[1],
                      kernel_size=config.conv_kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Flattened size after convolution
        dummy = torch.zeros(1, *config.input_shape)
        encoded = self.encoder(dummy)
        self.flattened_size = encoded.view(1, -1).size(1)

        # Latent projection
        self.fc_enc = nn.Sequential(
            nn.Linear(self.flattened_size, config.latent_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Decoder
        self.fc_dec = nn.Sequential(
            nn.Linear(config.latent_dim, self.flattened_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.flattened_size, c * h * w),
            nn.Sigmoid(),  # assume inputs in [0,1]
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map input image to latent vector."""
        x = self.encoder(inputs)
        x = x.view(x.size(0), -1)
        return self.fc_enc(x)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct image from latent vector."""
        x = self.fc_dec(latents)
        return x.view(-1, *self.input_shape)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def HybridAutoencoder(
    input_shape: Tuple[int, int, int],
    *,
    latent_dim: int = 32,
    conv_hidden_channels: Tuple[int, int] = (8, 16),
    dropout: float = 0.1,
) -> HybridAutoencoderNet:
    """Factory that returns a configured hybrid autoencoder."""
    config = HybridAutoencoderConfig(
        input_shape=input_shape,
        latent_dim=latent_dim,
        conv_hidden_channels=conv_hidden_channels,
        dropout=dropout,
    )
    return HybridAutoencoderNet(config)


def train_hybrid_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Train the hybrid autoencoder and return loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for epoch in range(epochs):
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


__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderConfig",
    "HybridAutoencoderNet",
    "train_hybrid_autoencoder",
]
