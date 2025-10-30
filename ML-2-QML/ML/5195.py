"""Hybrid classical network combining convolution, autoencoding, fully‑connected, and regression components.

The module exposes a single factory function `FCL()` that returns an instance of
`HybridFCL`.  The class implements a `forward` method that processes the input
through a 2×2 convolution, a fully‑connected auto‑encoder, a linear layer, and
finally a small regression network.  The implementation is fully
PyTorch‑compatible and can be integrated into standard training loops.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

# --------------------------------------------------------------------------- #
# Auto‑Encoder utilities (adapted from the Autoencoder reference)
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        encoder = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder.append(nn.Linear(in_dim, hidden))
            encoder.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder.append(nn.Linear(in_dim, hidden))
            decoder.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

def Autoencoder(input_dim: int, *, latent_dim: int = 32,
                hidden_dims: Tuple[int, int] = (128, 64),
                dropout: float = 0.1) -> AutoencoderNet:
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)

# --------------------------------------------------------------------------- #
# Convolution filter (adapted from Conv reference)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x of shape (batch, features) where features = kernel_size**2
        x = x.view(-1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(x)
        return torch.sigmoid(logits - self.threshold).mean(dim=(2, 3))

# --------------------------------------------------------------------------- #
# Estimator network (adapted from EstimatorQNN reference)
# --------------------------------------------------------------------------- #
class EstimatorNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --------------------------------------------------------------------------- #
# Hybrid network combining all components
# --------------------------------------------------------------------------- #
class HybridFCL(nn.Module):
    """
    A composite network that chains together a convolution filter, an
    auto‑encoder, a fully‑connected layer, and a small regression head.
    """
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=2)
        self.autoencoder = Autoencoder(input_dim, latent_dim=latent_dim,
                                       hidden_dims=hidden_dims, dropout=dropout)
        # Fully‑connected layer that accepts the concatenated conv output
        # and the auto‑encoder reconstruction.
        self.fcl = nn.Linear(input_dim + 1, 1)
        self.estimator = EstimatorNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, input_dim).  The first four elements are
            interpreted as a 2×2 patch for the convolution filter.
        """
        # Convolution filter
        conv_out = self.conv(x[:, :4])  # shape (batch, 1)
        # Auto‑encoder reconstruction
        z = self.autoencoder.encode(x)
        recon = self.autoencoder.decode(z)
        # Concatenate conv output with reconstruction
        combined = torch.cat([conv_out, recon], dim=1)  # shape (batch, input_dim+1)
        # Fully‑connected layer
        fcl_out = torch.tanh(self.fcl(combined))
        # Regression head
        out = self.estimator(fcl_out)
        return out

    def run(self, data: torch.Tensor | list | tuple) -> torch.Tensor:
        """Convenience wrapper that accepts NumPy arrays or lists."""
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)
        return self.forward(data)

def FCL() -> HybridFCL:
    """Factory that mirrors the quantum helper."""
    return HybridFCL(input_dim=4)

__all__ = ["FCL", "HybridFCL"]
