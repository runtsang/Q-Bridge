"""Autoencoder module with advanced training features.

Features
--------
- Flexible activation functions and optional batch‑norm.
- Optional skip connections between encoder and decoder.
- Early‑stopping and validation split.
- Training history returns both train and validation losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Callable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Optimizer


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
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU
    batch_norm: bool = False
    skip_connection: bool = False
    early_stop_patience: int | None = None
    device: torch.device | None = None


class AutoencoderNet(nn.Module):
    """Multilayer perceptron autoencoder with optional skip connections."""

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._build_network()

    def _build_network(self) -> None:
        enc_layers: list[nn.Module] = []
        in_dim = self.cfg.input_dim
        for h in self.cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            if self.cfg.batch_norm:
                enc_layers.append(nn.BatchNorm1d(h))
            enc_layers.append(self.cfg.activation())
            if self.cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(self.cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, self.cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = []
        in_dim = self.cfg.latent_dim
        for h in reversed(self.cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            if self.cfg.batch_norm:
                dec_layers.append(nn.BatchNorm1d(h))
            dec_layers.append(self.cfg.activation())
            if self.cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(self.cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, self.cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        if self.cfg.skip_connection:
            # Concatenate input with latent for richer decoding
            x = torch.cat([x, z], dim=1)
        return self.decode(z) if not self.cfg.skip_connection else self.decode(x)


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    val_split: float = 0.1,
    early_stop_patience: int | None = None,
    optimizer_cls: type[Optimizer] = torch.optim.Adam,
    loss_fn: nn.Module = nn.MSELoss(),
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """Train the autoencoder and return training history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = TensorDataset(_as_tensor(data))
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train": [], "val": []}
    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch, in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= n_train
        history["train"].append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch, in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = loss_fn(recon, batch)
                val_loss += loss.item() * batch.size(0)
        val_loss /= n_val
        history["val"].append(val_loss)

        if best_val > val_loss:
            best_val = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if early_stop_patience is not None and epochs_no_improve >= early_stop_patience:
            break

    return history


__all__ = ["AutoencoderConfig", "AutoencoderNet", "train_autoencoder"]
