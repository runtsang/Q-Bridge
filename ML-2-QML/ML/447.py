"""Enhanced PyTorch autoencoder with residual blocks, batch normalization, and early stopping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np


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
class AutoencoderConfig:
    """Configuration values for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    batchnorm: bool = True
    residual: bool = True
    early_stop_patience: int = 10
    lr_scheduler: bool = True


class AutoencoderNet(nn.Module):
    """A lightweight multilayer perceptron autoencoder with residuals and batch‑norm."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_block(config.input_dim, config.hidden_dims, is_encoder=True)
        self.latent_layer = nn.Linear(config.hidden_dims[-1], config.latent_dim)
        self.decoder = self._build_block(config.latent_dim, list(reversed(config.hidden_dims)), is_encoder=False)

    def _build_block(self, in_dim: int, dims: Tuple[int,...], is_encoder: bool) -> nn.Sequential:
        layers: List[nn.Module] = []
        prev_dim = in_dim
        for dim in dims:
            layers.append(nn.Linear(prev_dim, dim))
            if self.config.batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            if self.config.residual and prev_dim == dim:
                layers.append(ResidualAdd())
            prev_dim = dim
        return nn.Sequential(*layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.encode(inputs)
        return self.decode(latent)


class ResidualAdd(nn.Module):
    """Adds the input to the output of the previous layer if dimensions match."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    batchnorm: bool = True,
    residual: bool = True,
    early_stop_patience: int = 10,
    lr_scheduler: bool = True,
) -> AutoencoderNet:
    """Factory that returns a configured network."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        batchnorm=batchnorm,
        residual=residual,
        early_stop_patience=early_stop_patience,
        lr_scheduler=lr_scheduler,
    )
    return AutoencoderNet(config)


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    early_stop_patience: int | None = None,
) -> List[float]:
    """Training loop with early stopping and LR scheduler."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5, verbose=False) if model.config.lr_scheduler else None

    history: List[float] = []
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
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

        if scheduler:
            scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if early_stop_patience is not None and patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return history


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
]
