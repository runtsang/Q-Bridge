"""Hybrid autoencoder combining QCNN‑style feature extraction with a latent bottleneck."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


def _as_tensor(data: torch.Tensor | torch.Tensor) -> torch.Tensor:
    """Ensure data is a float32 tensor on the default device."""
    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data, dtype=torch.float32)
    else:
        data = data.to(dtype=torch.float32)
    return data


@dataclass
class HybridAutoencoderConfig:
    """Configuration for :class:`HybridAutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    max_qubit_dim: int = 8  # maximum QCNN qubit size


class HybridAutoencoderNet(nn.Module):
    """A hybrid autoencoder that uses QCNN‑style convolutional blocks for
    feature extraction before projecting to a latent space."""

    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # ---- Encoder: QCNN style blocks ----
        self.feature_map = nn.Sequential(nn.Linear(config.input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, config.latent_dim)

        # ---- Decoder: simple linear decoder ----
        decoder_layers = []
        in_dim = config.latent_dim
        hidden = config.hidden_dims
        for h in hidden:
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return a latent representation."""
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return self.head(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input from latent codes."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> HybridAutoencoderNet:
    """Factory mirroring the classical helper with a QCNN encoder."""
    cfg = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridAutoencoderNet(cfg)


def train_hybrid_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Simple reconstruction loop returning loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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


__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderNet",
    "HybridAutoencoderConfig",
    "train_hybrid_autoencoder",
]
