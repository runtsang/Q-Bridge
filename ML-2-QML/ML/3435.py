"""Combined classical kernel module with autoencoder latent embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _to_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Convert input to a float32 tensor on the current device."""
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


class AutoencoderNet(nn.Module):
    """Multilayer perceptron autoencoder with optional layer‑norm."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = self._make_mlp(
            config.input_dim,
            config.hidden_dims,
            config.latent_dim,
            config.dropout,
            encode=True,
        )
        self.decoder = self._make_mlp(
            config.latent_dim,
            config.hidden_dims[::-1],
            config.input_dim,
            config.dropout,
            encode=False,
        )

    @staticmethod
    def _make_mlp(
        in_dim: int,
        hidden_dims: Tuple[int,...],
        out_dim: int,
        dropout: float,
        encode: bool,
    ) -> nn.Sequential:
        layers = []
        current = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(current, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current = h
        layers.append(nn.Linear(current, out_dim))
        layers.append(nn.LayerNorm(out_dim))
        return nn.Sequential(*layers)

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
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Factory that returns a configurable autoencoder."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(cfg)


class RBFKernel(nn.Module):
    """Radial‑basis kernel operating on latent embeddings."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class QuantumKernelMethod(nn.Module):
    """
    Wrapper that first maps data through a classical autoencoder and then
    evaluates an RBF kernel on the resulting latent vectors.
    """

    def __init__(
        self,
        encoder: AutoencoderNet,
        kernel: RBFKernel,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.kernel = kernel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute kernel value between two input vectors."""
        z_x = self.encoder.encode(x)
        z_y = self.encoder.encode(y)
        return self.kernel(z_x, z_y).squeeze()


def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    encoder: AutoencoderNet,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Compute the Gram matrix between two datasets ``a`` and ``b`` after
    mapping them through ``encoder`` and applying an RBF kernel with
    width ``gamma``.
    """
    kernel = QuantumKernelMethod(encoder, RBFKernel(gamma))
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "RBFKernel",
    "QuantumKernelMethod",
    "kernel_matrix",
]
