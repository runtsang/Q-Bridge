"""
AutoencoderGen230: Classical MLP autoencoder module.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def _tensor_from(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Ensure input is a float32 torch.Tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    extra_layers: int = 0

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        for _ in range(config.extra_layers):
            encoder_layers.extend([nn.Linear(in_dim, in_dim), nn.ReLU()])
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        for _ in range(config.extra_layers):
            decoder_layers.extend([nn.Linear(in_dim, in_dim), nn.ReLU()])
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    extra_layers: int = 0,
) -> AutoencoderNet:
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        extra_layers=extra_layers,
    )
    return AutoencoderNet(cfg)

def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_tensor_from(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

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

__all__ = ["Autoencoder", "AutoencoderConfig", "AutoencoderNet", "train_autoencoder"]
