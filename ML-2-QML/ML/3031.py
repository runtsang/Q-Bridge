"""Hybrid autoencoder with classical quanvolution encoder.

This module merges the classical autoencoder from Autoencoder.py with
the convolutional feature extraction of QuanvolutionFilter.  The
encoder performs a 2‑pixel patch convolution followed by a fully
connected bottleneck.  The decoder reconstructs the image from the
latent vector.  The design is fully torch‑based and can be trained
with the provided `train_hybrid_auto` helper.

The class is intentionally lightweight so it can be dropped into
existing PyTorch pipelines or serve as a baseline for hybrid quantum
experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.nn import functional as F
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


class QuanvolutionFilter(nn.Module):
    """2×2 pixel patch convolution that mimics a quantum kernel.

    The filter is identical to the classical version from the original
    quanvolution example but written in pure PyTorch so it can be used
    inside a normal training loop.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x shape: (B, 1, H, W)
        features = self.conv(x)          # (B, 4, H/2, W/2)
        return features.view(x.size(0), -1)  # (B, 4*H/2*W/2)


@dataclass
class HybridAutoConfig:
    """Configuration for :class:`HybridAutoQuanvNet`."""
    input_shape: Tuple[int, int, int]  # (C, H, W)
    latent_dim: int = 32
    dropout: float = 0.1


class HybridAutoQuanvNet(nn.Module):
    """Hybrid classical autoencoder with a quanvolution encoder."""
    def __init__(self, config: HybridAutoConfig) -> None:
        super().__init__()
        self.config = config
        self.qfilter = QuanvolutionFilter()

        # Compute flattened size after the filter
        C, H, W = config.input_shape
        flat_size = 4 * (H // 2) * (W // 2)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, flat_size),
            nn.Sigmoid(),  # Assuming normalized pixel values
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent representation of input image."""
        features = self.qfilter(x)          # (B, flat_size)
        return self.encoder(features)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct image from latent vector."""
        flat = self.decoder(z)              # (B, flat_size)
        C, H, W = self.config.input_shape
        # reshape back to original image shape
        return flat.view(-1, C, H // 2, W // 2).reshape(-1, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


def train_hybrid_auto(
    model: HybridAutoQuanvNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop for the hybrid autoencoder."""
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
    "HybridAutoQuanvNet",
    "HybridAutoConfig",
    "train_hybrid_auto",
]
