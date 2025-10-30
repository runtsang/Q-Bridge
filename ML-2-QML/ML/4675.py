"""Hybrid classical autoencoder combining QCNN style encoder and MLP decoder."""

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


class QCNNModel(nn.Module):
    """Classical fully‑connected network inspired by the QCNN helper."""
    def __init__(self, input_dim: int, hidden_dims: Tuple[int,...] = (128, 64), dropout: float = 0.1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.Tanh()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


@dataclass
class HybridAutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    n_qubits: int = 0  # placeholder for quantum side


class HybridAutoencoder(nn.Module):
    """Hybrid autoencoder with a QCNN encoder, latent projection, and MLP decoder."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = QCNNModel(config.input_dim, config.hidden_dims, config.dropout)
        self.latent_proj = nn.Linear(config.hidden_dims[-1], config.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[-1], config.input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.latent_proj(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def HybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1
) -> HybridAutoencoder:
    config = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridAutoencoder(config)


def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
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


__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderFactory",
    "train_hybrid_autoencoder",
    "HybridAutoencoderConfig",
]
