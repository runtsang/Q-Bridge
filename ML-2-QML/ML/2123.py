"""Hybrid classical autoencoder with residual connections, early stopping, and learning‑rate scheduling.

The module defines :class:`AutoencoderHybrid`, a fully‑connected autoencoder that
extends the original seed by adding residual blocks and an optional early‑stopping
mechanism.  A lightweight training helper :func:`train_autoencoder` is provided
which supports cosine or plateau learning‑rate schedules.

The public API mirrors the anchor reference – a factory function :func:`Autoencoder`
creates a network instance from :class:`AutoencoderConfig`, and
:func:`train_autoencoder` returns the loss history.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
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
@dataclass
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderHybrid`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    lr: float = 1e-3
    patience: int = 5
    schedule: str = "cosine"  # options: "cosine", "plateau"

# --------------------------------------------------------------------------- #
# Residual block
# --------------------------------------------------------------------------- #
class _ResidualBlock(nn.Module):
    """Two‑layer residual block used in the encoder/decoder."""
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.dropout(out)
        return out + x

# --------------------------------------------------------------------------- #
# Autoencoder
# --------------------------------------------------------------------------- #
class AutoencoderHybrid(nn.Module):
    """Classical auto‑encoder with residual connections and optional early‑stopping."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    # ----------------------------------------------------------------------- #
    def _build_encoder(self) -> nn.Module:
        layers: List[nn.Module] = []
        in_dim = self.config.input_dim
        for h in self.config.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            layers.append(_ResidualBlock(h, self.config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, self.config.latent_dim))
        return nn.Sequential(*layers)

    # ----------------------------------------------------------------------- #
    def _build_decoder(self) -> nn.Module:
        layers: List[nn.Module] = []
        in_dim = self.config.latent_dim
        for h in reversed(self.config.hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            layers.append(_ResidualBlock(h, self.config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, self.config.input_dim))
        return nn.Sequential(*layers)

    # ----------------------------------------------------------------------- #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    # ----------------------------------------------------------------------- #
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    # ----------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderHybrid:
    """Return a configured :class:`AutoencoderHybrid`."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderHybrid(config)

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_autoencoder(
    model: AutoencoderHybrid,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float | None = None,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    use_scheduler: bool = True,
    val_data: Optional[torch.Tensor] = None,
) -> List[float]:
    """Train the auto‑encoder and return the training loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    lr = lr if lr is not None else model.config.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    scheduler = None
    if use_scheduler:
        if model.config.schedule == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        elif model.config.schedule == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    history: List[float] = []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
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

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(epoch_loss)
            else:
                scheduler.step()

        # Early stopping on validation set
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(_as_tensor(val_data).to(device))
                val_loss = loss_fn(val_pred, _as_tensor(val_data).to(device)).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= model.config.patience:
                break

    return history

__all__ = [
    "Autoencoder",
    "AutoencoderHybrid",
    "AutoencoderConfig",
    "train_autoencoder",
]
