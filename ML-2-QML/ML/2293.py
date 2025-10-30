"""Hybrid classical autoencoder with regression head."""

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
class HybridConfig:
    """Configuration for HybridAutoencoderEstimator."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    reg_hidden_dims: Tuple[int,...] = (8, 4)


class HybridAutoencoderEstimator(nn.Module):
    """Classical autoencoder with a regression head."""
    def __init__(self, config: HybridConfig) -> None:
        super().__init__()
        # Encoder
        enc_layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                enc_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                dec_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Regression head
        reg_layers = []
        in_dim = config.latent_dim
        for h in config.reg_hidden_dims:
            reg_layers.append(nn.Linear(in_dim, h))
            reg_layers.append(nn.Tanh())
            in_dim = h
        reg_layers.append(nn.Linear(in_dim, 1))
        self.regressor = nn.Sequential(*reg_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.regressor(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        pred = self.predict(z)
        return recon, pred


def train_hybrid(
    model: HybridAutoencoderEstimator,
    data: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Train reconstruction + regression loss."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data), _as_tensor(targets))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    history = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, pred = model(batch_x)
            loss = mse(recon, batch_x) + l1(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = ["HybridAutoencoderEstimator", "HybridConfig", "train_hybrid"]
