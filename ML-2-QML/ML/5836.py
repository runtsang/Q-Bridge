"""Hybrid classical auto‑encoder with integrated regression head.

This module extends the simple MLP autoencoder by adding a regression
head that operates on the latent representation.  It combines the
architecture of the original Autoencoder.py with the EstimatorQNN
regressor, allowing simultaneous reconstruction and prediction in a
single forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Helper to convert inputs to float32 tensors on the default device
def _as_tensor(data: torch.Tensor | list[float] | tuple[float,...]) -> torch.Tensor:
    tensor = data if isinstance(data, torch.Tensor) else torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)

@dataclass
class HybridAutoEncoderConfig:
    """Configuration for the hybrid auto‑encoder + regressor."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    regressor_dims: Tuple[int,...] = (8, 4)   # dims of the regression head

class HybridAutoEncoder(nn.Module):
    """A multi‑task MLP that reconstructs the input and predicts a scalar target."""
    def __init__(self, cfg: HybridAutoEncoderConfig) -> None:
        super().__init__()
        # Encoder
        enc_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if cfg.dropout:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if cfg.dropout:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Regression head
        reg_layers = []
        in_dim = cfg.latent_dim
        for h in cfg.regressor_dims:
            reg_layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        reg_layers.append(nn.Linear(in_dim, 1))
        self.regressor = nn.Sequential(*reg_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, prediction)."""
        z = self.encode(x)
        return self.decode(z), self.regressor(z)

def HybridAutoEncoderFactory(
    input_dim: int, *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    regressor_dims: Tuple[int,...] = (8, 4)
) -> HybridAutoEncoder:
    cfg = HybridAutoEncoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        regressor_dims=regressor_dims
    )
    return HybridAutoEncoder(cfg)

def train_hybrid_autoencoder(
    model: HybridAutoEncoder,
    data: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None
) -> list[float]:
    """Joint training loop that optimises reconstruction and regression losses."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data), _as_tensor(targets))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    recon_loss_fn = nn.MSELoss()
    reg_loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, pred = model(x_batch)
            recon_loss = recon_loss_fn(recon, x_batch)
            reg_loss = reg_loss_fn(pred, y_batch)
            loss = recon_loss + reg_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = ["HybridAutoEncoder", "HybridAutoEncoderConfig", "HybridAutoEncoderFactory", "train_hybrid_autoencoder"]
