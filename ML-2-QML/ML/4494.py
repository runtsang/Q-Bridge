"""
AutoencoderHybrid – classical implementation.

This module implements a fully‑connected autoencoder with an optional
sequence‑processing LSTM layer.  The class is designed to be
drop‑in compatible with the quantum version; it exposes the same
constructor signature and method names so that the same experiment
scripts can run on either backend.

The architecture is a direct evolution of the seed Autoencoder.py:
- Encoder/decoder stacks with ReLU and dropout.
- Optional LSTM wrapper that operates on the latent space.
- A lightweight `reconstruction_loss` helper for quick prototyping.

The code is intentionally free of any quantum dependencies and
provides a clean interface for future extensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: torch.Tensor | Iterable[float]) -> torch.Tensor:
    """Ensure input is a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclass
class AutoencoderConfig:
    """Configuration for the classical autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1


class ClassicalAutoencoder(nn.Module):
    """Fully‑connected encoder/decoder pair."""

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        # ---------- Encoder ----------
        enc_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # ---------- Decoder ----------
        dec_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class AutoencoderHybrid:
    """
    Classical autoencoder with an optional LSTM that consumes the latent
    representation.  The class signature matches the quantum variant so
    that experiment scripts can switch backends by simply importing
    the other module.
    """

    def __init__(
        self,
        cfg: AutoencoderConfig,
        *,
        use_lstm: bool = False,
        lstm_hidden: int = 64,
    ) -> None:
        self.model = ClassicalAutoencoder(cfg)
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=cfg.latent_dim,
                hidden_size=lstm_hidden,
                batch_first=True,
            )
            self.lstm_out = nn.Linear(lstm_hidden, cfg.latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.model.encoder(x)
        if self.use_lstm:
            z_seq, _ = self.lstm(z.unsqueeze(1))  # (B,1,D) → (B,1,D)
            z = self.lstm_out(z_seq.squeeze(1))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def reconstruction_loss(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(recon, x, reduction="mean")


def train_autoencoder(
    hybrid: AutoencoderHybrid,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> List[float]:
    """Quick training loop that returns the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hybrid.to(device)
    dataset = TensorDataset(_as_tensor(data).to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(hybrid.parameters(), lr=lr)
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = hybrid(batch)
            loss = hybrid.reconstruction_loss(batch, recon)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "AutoencoderHybrid",
    "AutoencoderConfig",
    "train_autoencoder",
]
