"""Hybrid SamplerQNN combining classical autoencoder, kernel, and quantum sampler.

The module exposes SamplerQNNGen422, a torch.nn.Module that

* encodes raw inputs via a lightweight autoencoder
* feeds the latent vector to a quantum sampler implemented in the
  companion QML module
* optionally applies a radial‑basis kernel to latent representations
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple, Sequence, Iterable

# ---------- Autoencoder ----------
class AutoencoderConfig:
    input_dim: int
    latent_dim: int
    hidden_dims: Tuple[int,...]
    dropout: float

    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int,...] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout


class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*layers)

        layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    config = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(config)


# ---------- Classical kernel ----------
class RBFKernel(nn.Module):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> torch.Tensor:
    k = RBFKernel(gamma)
    return torch.stack([k(a_i, b_i) for a_i, b_i in zip(a, b)])


# ---------- Quantum sampler ----------
# Import the quantum wrapper from the companion module.
try:
    from.SamplerQNN__gen422_qml import QuantumSampler
except Exception:  # pragma: no cover
    QuantumSampler = None  # type: ignore


class SamplerQNNGen422(nn.Module):
    """Hybrid sampler that uses an autoencoder to produce a latent vector,
    then forwards that vector to a quantum sampler.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int,...] = (128, 64),
        dropout: float = 0.1,
        kernel_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.autoencoder = Autoencoder(
            input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.kernel = RBFKernel(kernel_gamma)
        # The quantum sampler is lazily constructed once imported
        self.quantum_sampler = QuantumSampler() if QuantumSampler else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.encode(x)
        if self.quantum_sampler is None:
            raise RuntimeError(
                "QuantumSampler could not be imported. "
                "Ensure the companion QML module is available."
            )
        probs = self.quantum_sampler(z)
        return probs

    def kernel_sim(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return RBF kernel similarity for two batches of latent vectors."""
        return self.kernel(a, b)


__all__ = [
    "AutoencoderConfig",
    "AutoencoderNet",
    "Autoencoder",
    "RBFKernel",
    "kernel_matrix",
    "SamplerQNNGen422",
]
