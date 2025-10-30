"""Hybrid classical autoencoder with optional convolutional front‑end.

The module mirrors the original `Autoencoder.py` but extends it with:
* a convolutional encoder (inspired by `QuanvolutionFilter` and `QFCModel`);
* a regression dataset generator (from `QuantumRegression.py`);
* a unified factory and training routine that work with either FC or CNN
  encoders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset


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
    """Configuration values for :class:`AutoencoderHybrid`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    conv: bool = False  # use a 2‑D convolutional encoder
    conv_channels: int = 8  # number of output channels in the conv encoder


class AutoencoderHybrid(nn.Module):
    """Hybrid autoencoder with optional convolutional encoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder
        if config.conv:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, config.conv_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(config.conv_channels, config.conv_channels * 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(config.conv_channels * 2 * 7 * 7, config.hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dims[0], config.latent_dim),
            )
        else:
            layers = []
            in_dim = config.input_dim
            for hidden in config.hidden_dims:
                layers.append(nn.Linear(in_dim, hidden))
                layers.append(nn.ReLU())
                if config.dropout > 0.0:
                    layers.append(nn.Dropout(config.dropout))
                in_dim = hidden
            layers.append(nn.Linear(in_dim, config.latent_dim))
            self.encoder = nn.Sequential(*layers)

        # Decoder
        layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*layers)

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
    conv: bool = False,
    conv_channels: int = 8,
) -> AutoencoderHybrid:
    """Factory that mirrors the quantum helper returning a configured network."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        conv=conv,
        conv_channels=conv_channels,
    )
    return AutoencoderHybrid(config)


def train_autoencoder(
    model: AutoencoderHybrid,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Simple reconstruction training loop returning the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
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
    return history


# --------------------------------------------------------------------------- #
# Dataset utilities (borrowed from QuantumRegression.py)
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_features: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic regression data with a sinusoidal target."""
    x = torch.rand(samples, num_features, dtype=torch.float32) * 2 - 1
    angles = x.sum(dim=1)
    y = torch.sin(angles) + 0.1 * torch.cos(2 * angles)
    return x, y


class RegressionDataset(Dataset):
    """Simple regression dataset with synthetic superposition data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {"states": self.features[index], "target": self.labels[index]}


__all__ = [
    "AutoencoderHybrid",
    "Autoencoder",
    "train_autoencoder",
    "RegressionDataset",
    "generate_superposition_data",
]
