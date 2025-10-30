"""Hybrid regression model combining a classical autoencoder and a linear regression head.

The module preserves the original regression dataset and training utilities
while adding an autoencoder that learns a latent representation before
the final regression layer.  The autoencoder is trained jointly with the
regression loss, providing a richer feature space for the linear head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data in the form of a noisy sinusoid.

    Parameters
    ----------
    num_features: int
        Dimensionality of the input feature vector.
    samples: int
        Number of samples to generate.

    Returns
    -------
    x: np.ndarray of shape (samples, num_features)
        Uniformly distributed features in [-1, 1].
    y: np.ndarray of shape (samples,)
        Target values computed as ``sin(sum(x)) + 0.1*cos(2*sum(x))``.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset wrapper around the synthetic superposition data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
#  Autoencoder utilities – the classical building block
# --------------------------------------------------------------------------- #

class AutoencoderConfig:
    """Configuration holder for :class:`AutoencoderNet`."""

    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout


class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder with configurable depth and dropout."""

    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        # Encoder
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

        # Decoder
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Convenience factory mirroring the quantum helper."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(cfg)


# --------------------------------------------------------------------------- #
#  Hybrid regression model – classical + autoencoder
# --------------------------------------------------------------------------- #

class HybridRegressionModel(nn.Module):
    """Regression head that operates on a latent representation learned by an autoencoder.

    The model can be trained jointly: the autoencoder learns to reconstruct
    the input while the linear head learns to predict the target.
    """

    def __init__(self, num_features: int, latent_dim: int = 32):
        super().__init__()
        self.autoencoder = Autoencoder(num_features, latent_dim=latent_dim)
        self.regressor = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.autoencoder.encode(x)
        return self.regressor(latent).squeeze(-1)


# Backward‑compatibility alias used in the original anchor
QModel = HybridRegressionModel

__all__ = [
    "HybridRegressionModel",
    "RegressionDataset",
    "generate_superposition_data",
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "QModel",
]
