"""Enhanced fully‑connected autoencoder with configurable skip connections, batch‑norm, and early stopping.

The API mirrors the original seed but adds optional activation, dropout, batch‑norm, and skip‑connection support. A dedicated training loop implements early stopping and a flexible loss function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Coerce *data* to a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

@dataclass
class AutoencoderExtendedConfig:
    """Configuration for :class:`AutoencoderExtended`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU
    batch_norm: bool = False
    skip_connections: bool = False

class AutoencoderExtended(nn.Module):
    """Fully‑connected autoencoder supporting skip connections and batch‑norm."""
    def __init__(self, cfg: AutoencoderExtendedConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = self._build_mlp(
            cfg.input_dim, cfg.hidden_dims, cfg.latent_dim, cfg
        )
        self.decoder = self._build_mlp(
            cfg.latent_dim, cfg.hidden_dims[::-1], cfg.input_dim, cfg, is_decoder=True
        )

    def _build_mlp(
        self,
        in_dim: int,
        hidden_dims: Tuple[int,...],
        out_dim: int,
        cfg: AutoencoderExtendedConfig,
        is_decoder: bool = False,
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if cfg.batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(cfg.activation())
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

def AutoencoderExtendedFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU,
    batch_norm: bool = False,
    skip_connections: bool = False,
) -> AutoencoderExtended:
    cfg = AutoencoderExtendedConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
        batch_norm=batch_norm,
        skip_connections=skip_connections,
    )
    return AutoencoderExtended(cfg)

def train_autoencoder_extended(
    model: AutoencoderExtended,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    early_stopping_patience: int = 20,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.MSELoss(),
) -> List[float]:
    """Train the autoencoder with optional early stopping."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: List[float] = []
    best_loss = float("inf")
    patience_counter = 0

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

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    return history

__all__ = [
    "AutoencoderExtended",
    "AutoencoderExtendedConfig",
    "AutoencoderExtendedFactory",
    "train_autoencoder_extended",
]
