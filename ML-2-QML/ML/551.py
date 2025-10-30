"""PyTorch implementation of an extended autoencoder with skip connections and batch normalization."""

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
class AutoencoderConfig:
    """Configuration values for :class:`AutoencoderHybridNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    batch_norm: bool = False
    skip_connections: bool = False


class AutoencoderHybridNet(nn.Module):
    """An autoencoder with optional batch‑norm and skip connections."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        # Build encoder
        self.encoder = nn.ModuleList()
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            block = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
            )
            if config.batch_norm:
                block.append(nn.BatchNorm1d(hidden))
            if config.dropout > 0.0:
                block.append(nn.Dropout(config.dropout))
            self.encoder.append(block)
            in_dim = hidden
        self.latent_layer = nn.Linear(in_dim, config.latent_dim)

        # Build decoder
        self.decoder = nn.ModuleList()
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            block = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
            )
            if config.batch_norm:
                block.append(nn.BatchNorm1d(hidden))
            if config.dropout > 0.0:
                block.append(nn.Dropout(config.dropout))
            self.decoder.append(block)
            in_dim = hidden
        self.output_layer = nn.Linear(in_dim, config.input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x)
        return self.latent_layer(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder:
            z = layer(z)
        return self.output_layer(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.encode(x)
        recon = self.decode(latent)
        if self.config.skip_connections:
            recon += x  # simple residual
        return recon


def AutoencoderHybrid(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    batch_norm: bool = False,
    skip_connections: bool = False,
) -> AutoencoderHybridNet:
    """Factory that returns a configured :class:`AutoencoderHybridNet`."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        batch_norm=batch_norm,
        skip_connections=skip_connections,
    )
    return AutoencoderHybridNet(config)


def train_autoencoder(
    model: AutoencoderHybridNet,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    early_stop_patience: int | None = None,
) -> list[float]:
    """Training loop with optional early stopping."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []
    best_loss = float("inf")
    counter = 0
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
        if early_stop_patience is not None:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                counter = 0
            else:
                counter += 1
            if counter >= early_stop_patience:
                break
    return history


__all__ = [
    "AutoencoderHybrid",
    "AutoencoderHybridNet",
    "AutoencoderConfig",
    "train_autoencoder",
]
