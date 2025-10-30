"""Hybrid classical autoencoder with 200‑epoch training and EstimatorQNN‑inspired decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

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
    """Configuration values for :class:`HybridAutoencoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    decoder_hidden: Tuple[int, int] = (8, 4)
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3


class HybridAutoencoder(nn.Module):
    """Classical autoencoder with EstimatorQNN‑inspired decoder."""

    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            *self._build_mlp(
                in_dim=config.input_dim,
                hidden_dims=config.hidden_dims,
                out_dim=config.latent_dim,
                dropout=config.dropout,
            )
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.decoder_hidden[0]),
            nn.Tanh(),
            nn.Linear(config.decoder_hidden[0], config.decoder_hidden[1]),
            nn.Tanh(),
            nn.Linear(config.decoder_hidden[1], config.input_dim),
        )

    @staticmethod
    def _build_mlp(
        in_dim: int,
        hidden_dims: Tuple[int,...],
        out_dim: int,
        dropout: float,
    ) -> List[nn.Module]:
        layers: List[nn.Module] = []
        current = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(current, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current = h
        layers.append(nn.Linear(current, out_dim))
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)


def HybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    decoder_hidden: Tuple[int, int] = (8, 4),
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> HybridAutoencoder:
    """Return a configured :class:`HybridAutoencoder` instance."""
    config = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        decoder_hidden=decoder_hidden,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    return HybridAutoencoder(config)


def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Train the hybrid autoencoder and return loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    epochs = epochs or model.decoder[-1].out_features  # fallback to config epochs
    batch_size = batch_size or 64
    lr = lr or 1e-3

    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

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
]
