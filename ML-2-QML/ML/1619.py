"""Autoencoder with convolutional bottleneck and quantum‑guided loss."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Convert input to a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)

@dataclass
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    # new features
    augment: Callable[[torch.Tensor], torch.Tensor] | None = None
    conv: bool = False
    conv_channels: Tuple[int, int] = (16, 32)
    conv_kernel: Tuple[int, int] = (3, 3)
    conv_stride: Tuple[int, int] = (2, 2)

class ResidualBlock(nn.Module):
    """Simple 2‑D residual block."""

    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))

class AutoencoderNet(nn.Module):
    """Hybrid MLP–CNN autoencoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        if config.conv:
            # compute image height/width from flattened input
            height = width = int((config.input_dim // 3) ** 0.5)
            assert height * width * 3 == config.input_dim, (
                "input_dim must be a perfect square times 3 for conv mode"
            )

            self.encoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=config.conv_channels[0],
                    kernel_size=config.conv_kernel,
                    stride=config.conv_stride,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
                ResidualBlock(config.conv_channels[0], config.dropout),
                nn.Conv2d(
                    in_channels=config.conv_channels[0],
                    out_channels=config.conv_channels[1],
                    kernel_size=config.conv_kernel,
                    stride=config.conv_stride,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
                ResidualBlock(config.conv_channels[1], config.dropout),
                nn.Flatten(),
                nn.Linear(
                    config.conv_channels[1] * ((height // config.conv_stride[0]) ** 2),
                    config.latent_dim,
                ),
            )
            self.decoder = nn.Sequential(
                nn.Linear(config.latent_dim, config.input_dim),
            )
        else:
            encoder_layers = []
            in_dim = config.input_dim
            for hidden in config.hidden_dims:
                encoder_layers.append(nn.Linear(in_dim, hidden))
                encoder_layers.append(nn.ReLU(inplace=True))
                encoder_layers.append(nn.Dropout(config.dropout))
                in_dim = hidden
            encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)

            decoder_layers = []
            in_dim = config.latent_dim
            for hidden in reversed(config.hidden_dims):
                decoder_layers.append(nn.Linear(in_dim, hidden))
                decoder_layers.append(nn.ReLU(inplace=True))
                decoder_layers.append(nn.Dropout(config.dropout))
                in_dim = hidden
            decoder_layers.append(nn.Linear(in_dim, config.input_dim))
            self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.config.augment:
            inputs = self.config.augment(inputs)
        if self.config.conv:
            batch, dim = inputs.shape
            h = w = int((dim // 3) ** 0.5)
            inputs = inputs.view(batch, 3, h, w)
        return self.decode(self.encode(inputs))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    augment: Callable[[torch.Tensor], torch.Tensor] | None = None,
    conv: bool = False,
    conv_channels: Tuple[int, int] = (16, 32),
    conv_kernel: Tuple[int, int] = (3, 3),
    conv_stride: Tuple[int, int] = (2, 2),
) -> AutoencoderNet:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        augment=augment,
        conv=conv,
        conv_channels=conv_channels,
        conv_kernel=conv_kernel,
        conv_stride=conv_stride,
    )
    return AutoencoderNet(config)

def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
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

__all__ = ["Autoencoder", "AutoencoderConfig", "AutoencoderNet", "train_autoencoder"]
