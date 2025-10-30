"""Hybrid classical autoencoder with a quantum‑inspired latent layer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple

# --------------------------------------------------------------------------- #
# Classical Sampler network mirroring a quantum SamplerQNN
# --------------------------------------------------------------------------- #
class SamplerQNN(nn.Module):
    """A lightweight neural network that emulates a quantum sampler."""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 4, output_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(x), dim=-1)

# --------------------------------------------------------------------------- #
# Configuration dataclass
# --------------------------------------------------------------------------- #
@dataclass
class HybridAutoencoderConfig:
    """Configuration for :class:`HybridAutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    quantum: bool = False  # whether to use quantum‑inspired sampler

# --------------------------------------------------------------------------- #
# Hybrid autoencoder implementation
# --------------------------------------------------------------------------- #
class HybridAutoencoderNet(nn.Module):
    """A hybrid autoencoder that optionally replaces the latent layer with a quantum sampler."""
    def __init__(self, cfg: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = self._build_mlp(cfg.input_dim, cfg.hidden_dims, cfg.latent_dim, cfg.dropout)
        self.quantum_layer = SamplerQNN(input_dim=cfg.latent_dim, hidden_dim=cfg.latent_dim, output_dim=cfg.latent_dim) if cfg.quantum else None
        self.decoder = self._build_mlp(cfg.latent_dim, tuple(reversed(cfg.hidden_dims)), cfg.input_dim, cfg.dropout)

    @staticmethod
    def _build_mlp(in_dim: int, hidden_dims: Tuple[int,...], out_dim: int, dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def quantum_transform(self, z: torch.Tensor) -> torch.Tensor:
        if self.quantum_layer is None:
            return z
        return self.quantum_layer(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        z = self.quantum_transform(z)
        return self.decode(z)

# --------------------------------------------------------------------------- #
# Factory helper
# --------------------------------------------------------------------------- #
def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    quantum: bool = False,
) -> HybridAutoencoderNet:
    cfg = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        quantum=quantum,
    )
    return HybridAutoencoderNet(cfg)

# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def _as_tensor(data: torch.Tensor | list[float]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

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
    "HybridAutoencoderConfig",
    "HybridAutoencoderNet",
    "SamplerQNN",
    "train_hybrid_autoencoder",
]
