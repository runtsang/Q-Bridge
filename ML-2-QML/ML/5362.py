"""Hybrid kernel implementation for classical machine learning.

The class implements a radial‑basis‑function (RBF) kernel that can optionally
pre‑process data with a quanvolution filter and compress it through a
fully‑connected autoencoder.  All components are pure PyTorch and
compatible with GPU acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# --------------------------------------------------------------------------- #
# Autoencoder utilities
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Simple MLP autoencoder."""

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*layers)

        # Decoder
        layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class Autoencoder:
    """Convenience factory mirroring the quantum helper."""

    def __init__(self, input_dim: int, *, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1) -> None:
        cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
        self.model = AutoencoderNet(cfg)


# --------------------------------------------------------------------------- #
# Quanvolution filter
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    """2‑pixel patch encoder using a 4‑qubit quantum kernel."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.conv(x).view(x.size(0), -1)


# --------------------------------------------------------------------------- #
# Hybrid kernel
# --------------------------------------------------------------------------- #
class HybridKernel(nn.Module):
    """Classical RBF kernel with optional quanvolution + autoencoder preprocessing."""

    def __init__(self,
                 gamma: float = 1.0,
                 use_autoencoder: bool = False,
                 autoencoder_cfg: AutoencoderConfig | None = None,
                 use_quanvolution: bool = False) -> None:
        super().__init__()
        self.gamma = gamma
        self.use_autoencoder = use_autoencoder
        self.use_quanvolution = use_quanvolution

        if self.use_autoencoder:
            if autoencoder_cfg is None:
                raise ValueError("autoencoder_cfg must be provided when use_autoencoder=True")
            self.autoencoder = AutoencoderNet(autoencoder_cfg)
        if self.use_quanvolution:
            self.quanvolution = QuanvolutionFilter()

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quanvolution:
            x = self.quanvolution(x)
        if self.use_autoencoder:
            x = self.autoencoder.encode(x)
        return x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._preprocess(x)
        y = self._preprocess(y)
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between two datasets."""
        mat = [[self.forward(x, y).item() for y in b] for x in a]
        return np.array(mat)


__all__ = ["HybridKernel", "AutoencoderConfig", "AutoencoderNet", "Autoencoder", "QuanvolutionFilter"]
