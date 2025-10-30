"""Hybrid classical-quantum autoencoder that fuses convolutional encoding,
a quantum-inspired latent layer, and a fully connected decoder.

The architecture mirrors the classical Autoencoder while adding a
parameterised “quantum” transformation that can be replaced by an actual
quantum circuit during experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Optional: import the classical Conv filter if available
try:
    from Conv import Conv  # type: ignore
except Exception:  # pragma: no cover
    Conv = None


def _as_tensor(data: torch.Tensor | Iterable[float]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)


@dataclass
class AutoencoderHybridConfig:
    """Configuration for :class:`AutoencoderHybridNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    conv_channels: int = 8  # number of channels in the conv encoder


class QuantumInspiredLayer(nn.Module):
    """A classical surrogate for a quantum layer using sin‑activation."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.linear(x))


class AutoencoderHybridNet(nn.Module):
    """Convolutional encoder → quantum‑inspired latent → fully‑connected decoder."""
    def __init__(self, config: AutoencoderHybridConfig) -> None:
        super().__init__()
        self.config = config

        # Convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, config.conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(config.conv_channels, config.conv_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Size of the flattened feature map after pooling
        conv_out = config.conv_channels * 2 * (config.input_dim // 4) ** 2

        # Quantum‑inspired latent layer
        self.quantum_latent = QuantumInspiredLayer(conv_out, config.latent_dim)

        # Fully‑connected decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, conv_out),
            nn.ReLU(),
            nn.Unflatten(1, (config.conv_channels * 2, config.input_dim // 4, config.input_dim // 4)),
            nn.ConvTranspose2d(config.conv_channels * 2, config.conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(config.conv_channels, 1, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation of *x*."""
        return self.quantum_latent(self.encoder(x).flatten(start_dim=1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input from latent vector *z*."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """End‑to‑end reconstruction."""
        return self.decode(self.encode(x))


def AutoencoderHybrid(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    conv_channels: int = 8,
) -> AutoencoderHybridNet:
    """Factory mirroring the original Autoencoder API."""
    config = AutoencoderHybridConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        conv_channels=conv_channels,
    )
    return AutoencoderHybridNet(config)


def train_autoencoder(
    model: AutoencoderHybridNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop that returns a list of epoch‑wise MSE losses."""
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


__all__ = [
    "AutoencoderHybrid",
    "AutoencoderHybridConfig",
    "AutoencoderHybridNet",
    "train_autoencoder",
]
