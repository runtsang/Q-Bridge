from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import mean_squared_error


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
    sparsity: float = 0.0  # fraction of units to drop in each hidden layer
    early_stop_patience: int = 10  # epochs to wait for validation improvement


class AutoencoderNet(nn.Module):
    """A lightweight MLP autoencoder with optional sparsity and early stopping."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_mlp(
            in_dim=config.input_dim,
            out_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            sparsity=config.sparsity,
        )
        self.decoder = self._build_mlp(
            in_dim=config.latent_dim,
            out_dim=config.input_dim,
            hidden_dims=list(reversed(config.hidden_dims)),
            dropout=config.dropout,
            sparsity=config.sparsity,
        )

    def _build_mlp(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Tuple[int,...],
        dropout: float,
        sparsity: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        curr_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            if sparsity > 0.0:
                # randomly mask a fraction of units
                mask = torch.rand(h) < sparsity
                layers.append(nn.Lambda(lambda x, m=mask: x * m))
            curr_dim = h
        layers.append(nn.Linear(curr_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    sparsity: float = 0.0,
    early_stop_patience: int = 10,
) -> AutoencoderNet:
    """Factory that mirrors the quantum helper returning a configured network."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        sparsity=sparsity,
        early_stop_patience=early_stop_patience,
    )
    return AutoencoderNet(config)


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    val_split: float = 0.1,
) -> tuple[list[float], list[float]]:
    """Simple reconstruction training loop returning loss histories for train/val."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    if val_split > 0:
        n_val = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
        loaders = {
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            "val": DataLoader(val_ds, batch_size=batch_size),
        }
    else:
        loaders = {"train": DataLoader(dataset, batch_size=batch_size, shuffle=True)}

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = {"train": [], "val": []}
    best_val = float("inf")
    patience = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch, in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(loaders["train"].dataset)
        history["train"].append(train_loss)

        if val_split > 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch, in loaders["val"]:
                    batch = batch.to(device)
                    recon = model(batch)
                    loss = loss_fn(recon, batch)
                    val_loss += loss.item() * batch.size(0)
            val_loss /= len(loaders["val"].dataset)
            history["val"].append(val_loss)

            # early stopping
            if val_loss < best_val:
                best_val = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= model.config.early_stop_patience:
                    warnings.warn(f"Early stopping after {epoch+1} epochs")
                    break

    return history["train"], history["val"]


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
]
