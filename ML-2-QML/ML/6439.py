"""Hybrid autoencoder with a classical MLP encoder/decoder and an optional ConvFilter layer.

The network preserves the public API of the original Autoencoder.py
while incorporating the Conv.py ConvFilter as a drop‑in pre‑processing
step.  The implementation is fully classical and can be trained
with the provided `train_autoencoder` helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple
import math

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
class AutoencoderConfig:
    """Configuration for :class:`HybridAutoencoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class ConvFilter(nn.Module):
    """Simple 2‑D convolution filter mimicking the classical Conv.py implementation."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply convolution, sigmoid activation and mean reduction."""
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(2, 3))


class HybridAutoencoder(nn.Module):
    """Classical MLP autoencoder with an optional ConvFilter encoder."""

    def __init__(self, config: AutoencoderConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        self.device = device or torch.device("cpu")

        # Optional ConvFilter pre‑processing
        self.sqrt_dim = int(math.sqrt(config.input_dim))
        if self.sqrt_dim ** 2!= config.input_dim:
            raise ValueError("input_dim must be a perfect square for the ConvFilter.")
        self.conv = ConvFilter(kernel_size=2)
        conv_out_dim = (self.sqrt_dim - 1) * (self.sqrt_dim - 1)

        # Encoder
        encoder_layers = [nn.Flatten()]
        in_dim = conv_out_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = [nn.Linear(config.latent_dim, config.hidden_dims[-1])]
        in_dim = config.hidden_dims[-1]
        for hidden in reversed(config.hidden_dims[:-1]):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        # Reshape to image, apply ConvFilter, then flatten
        batch = inputs.view(-1, 1, self.sqrt_dim, self.sqrt_dim)
        conv_out = self.conv(batch)
        flat = conv_out.view(conv_out.size(0), -1)
        return self.encoder(flat)

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
) -> HybridAutoencoder:
    """Factory that mirrors the quantum helper returning a configured network."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridAutoencoder(cfg)


def train_autoencoder(
    model: HybridAutoencoder,
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


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "HybridAutoencoder",
    "train_autoencoder",
]
