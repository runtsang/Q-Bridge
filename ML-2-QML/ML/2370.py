"""Classical quanvolution + autoencoder hybrid network.

This module defines a `QuanvolutionAutoencoder` that first extracts 2×2 patches
with a small Conv2d layer (the “quanvolution”) and then passes the
flattened feature vector to a configurable MLP autoencoder.  The design
mirrors the original seed but adds a flexible configuration API and a
training helper that returns a loss history.

The architecture is intentionally lightweight so that it can be used as a
drop‑in replacement for the original `QuanvolutionClassifier` in
benchmarks that require an encoder/decoder pair.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Iterable

# --------------------------------------------------------------------------- #
# 1. Classical quanvolution filter
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    """Convolutional patch extractor that emulates the original quanvolution."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: (batch, out_channels, H', W')
        feat = self.conv(x)
        # Flatten spatial dimensions
        return feat.view(x.size(0), -1)

# --------------------------------------------------------------------------- #
# 2. Configurable MLP autoencoder (borrowed from the seed)
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """A lightweight fully‑connected autoencoder."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = self._make_mlp(cfg.input_dim, cfg.hidden_dims,
                                      cfg.latent_dim, cfg.dropout)
        self.decoder = self._make_mlp(cfg.latent_dim,
                                      tuple(reversed(cfg.hidden_dims)),
                                      cfg.input_dim, cfg.dropout)

    @staticmethod
    def _make_mlp(in_dim: int, hidden: Tuple[int,...],
                  out_dim: int, dropout: float) -> nn.Sequential:
        layers = []
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

# --------------------------------------------------------------------------- #
# 3. Combined quanvolution + autoencoder
# --------------------------------------------------------------------------- #
class QuanvolutionAutoencoder(nn.Module):
    """Hybrid network that first applies a quanvolution filter and then
    a classical autoencoder on the flattened patch features."""
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 4,
                 patch_size: int = 2,
                 stride: int = 2,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels, out_channels,
                                          kernel_size=patch_size, stride=stride)
        # Compute the dimensionality of the input to the autoencoder
        img_size = 28  # MNIST assumption
        num_patches = (img_size // patch_size) ** 2
        input_dim = out_channels * num_patches
        cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
        self.autoencoder = AutoencoderNet(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the reconstructed image."""
        features = self.qfilter(x)
        recon = self.autoencoder(features)
        return recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation."""
        return self.autoencoder.encode(self.qfilter(x))

# --------------------------------------------------------------------------- #
# 4. Helper: tensor conversion
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

# --------------------------------------------------------------------------- #
# 5. Training routine
# --------------------------------------------------------------------------- #
def train_qa(model: nn.Module,
             data: torch.Tensor,
             *,
             epochs: int = 100,
             batch_size: int = 64,
             lr: float = 1e-3,
             device: torch.device | None = None) -> list[float]:
    """Simple reconstruction training loop that returns the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

__all__ = [
    "QuanvolutionFilter",
    "AutoencoderConfig",
    "AutoencoderNet",
    "QuanvolutionAutoencoder",
    "train_qa",
]
