"""Hybrid classical kernel with auto‑encoder feature extraction.

The :class:`AutoKernel` class combines a standard RBF kernel with a
trainable auto‑encoder that projects the input data into a low‑dimensional
latent space.  The kernel is computed on the latent vectors, yielding a
data‑adaptive similarity measure that can be used in kernel‑based
learning algorithms (SVM, Gaussian processes, etc.).

The class is fully compatible with the original ``Kernel`` API from the
anchor file: it exposes a :meth:`forward` method that accepts two tensors
and a :func:`kernel_matrix` helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------

@dataclass
class AutoencoderConfig:
    """Configuration for the internal auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    device: torch.device | None = None

# ---------------------------------------------------------------------------

class AutoencoderNet(nn.Module):
    """Simple fully‑connected auto‑encoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

# ---------------------------------------------------------------------------

class AutoKernel(nn.Module):
    """
    RBF kernel on the latent space produced by a trainable auto‑encoder.

    Parameters
    ----------
    gamma : float
        Width parameter of the RBF kernel.
    autoencoder_cfg : AutoencoderConfig
        Configuration for the internal auto‑encoder.
    """
    def __init__(self, gamma: float = 1.0,
                 autoencoder_cfg: AutoencoderConfig | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        if autoencoder_cfg is None:
            raise ValueError("autoencoder_cfg must be provided")
        self.autoencoder = AutoencoderNet(autoencoder_cfg)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel between two batches in latent space.

        Parameters
        ----------
        x, y : torch.Tensor
            Input tensors of shape (N, D) and (M, D) respectively.

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (N, M).
        """
        # Ensure inputs are 2‑D
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])

        # Encode into latent space
        z_x = self.autoencoder.encode(x)
        z_y = self.autoencoder.encode(y)

        # Compute squared Euclidean distances
        diff = z_x.unsqueeze(1) - z_y.unsqueeze(0)
        dist2 = torch.sum(diff * diff, dim=2)

        return torch.exp(-self.gamma * dist2)

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  gamma: float = 1.0,
                  autoencoder_cfg: AutoencoderConfig | None = None) -> np.ndarray:
    """
    Convenience wrapper to compute the Gram matrix for arbitrary
    sequences of tensors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of tensors representing two datasets.
    gamma : float
        RBF width.
    autoencoder_cfg : AutoencoderConfig
        Auto‑encoder configuration.

    Returns
    -------
    np.ndarray
        Kernel matrix of shape (len(a), len(b)).
    """
    kernel = AutoKernel(gamma, autoencoder_cfg)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["AutoencoderConfig", "AutoencoderNet",
           "AutoKernel", "kernel_matrix"]
