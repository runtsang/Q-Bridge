"""Hybrid autoencoder combining classical MLP, self‑attention, and quantum‑inspired FC layers."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable, Tuple
import numpy as np


# ---------- Helper modules ----------
class ClassicalSelfAttention(nn.Module):
    """A lightweight self‑attention block compatible with the quantum interface."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Treat the latent vector as a sequence of length 1
        q = x @ nn.Parameter(torch.randn(self.embed_dim, self.embed_dim))
        k = x @ nn.Parameter(torch.randn(self.embed_dim, self.embed_dim))
        v = x
        scores = nn.functional.softmax(q @ k.transpose(-1, -2) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v


class QuantumInspiredFC(nn.Module):
    """Classical surrogate of the Quantum‑NAT fully‑connected block."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
        )
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc(x)
        return self.bn(x)


# ---------- Core autoencoder ----------
class HybridAutoencoder(nn.Module):
    """Hybrid classical autoencoder integrating self‑attention and quantum‑inspired FC."""
    def __init__(self, config: "AutoencoderConfig") -> None:
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

        self.attention = ClassicalSelfAttention(embed_dim=config.latent_dim)

        self.qfc = QuantumInspiredFC(in_features=config.latent_dim, out_features=config.latent_dim)

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
        z = self.encoder(x)
        z = self.attention(z.unsqueeze(1)).squeeze(1)  # treat latent as sequence of length 1
        z = self.qfc(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


# ---------- Configuration ----------
class AutoencoderConfig:
    """Configuration for :class:`HybridAutoencoder`."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout


# ---------- Factory ----------
def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> HybridAutoencoder:
    """Return a configured hybrid autoencoder."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridAutoencoder(config)


# ---------- Training helper ----------
def train_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop that returns loss history."""
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


# ---------- Kernel utilities ----------
def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute RBF kernel matrix between two collections."""
    return np.array([[torch.exp(-gamma * torch.sum((x - y) ** 2)).item() for y in b] for x in a])


# ---------- Utility ----------
def _as_tensor(data: torch.Tensor | Iterable[float]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


__all__ = [
    "HybridAutoencoder",
    "Autoencoder",
    "AutoencoderConfig",
    "train_autoencoder",
    "kernel_matrix",
]
