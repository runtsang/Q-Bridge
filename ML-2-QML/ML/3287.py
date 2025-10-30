"""Hybrid QCNN–Autoencoder implementation in PyTorch.

The model follows the QCNN layer pattern (feature map → convolution → pooling)
but introduces an explicit bottleneck (latent_dim) and a symmetric decoder
to reconstruct the input.  It can be used as a classifier or as a feature
extractor for downstream tasks.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple


class QCNNAutoencoderModel(nn.Module):
    """
    A fully‑connected network that emulates the QCNN convolution and pooling
    layers as an encoder, followed by a symmetric decoder that reconstructs
    the input.  The encoder compresses to ``latent_dim`` features.
    """

    def __init__(
        self,
        input_dim: int = 8,
        latent_dim: int = 4,
        hidden_dims: Tuple[int, int] = (16, 8),
    ) -> None:
        super().__init__()
        # Encoder: feature_map → conv1 → pool1 → conv2 → pool2 → conv3
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], latent_dim),
        )

        # Decoder: mirrors the encoder in reverse order
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input from latent."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and immediately decode the input."""
        return self.decode(self.encode(x))


def QCNNAutoencoder(
    input_dim: int = 8,
    latent_dim: int = 4,
    hidden_dims: Tuple[int, int] = (16, 8),
) -> QCNNAutoencoderModel:
    """
    Factory function mirroring the classical QCNN helper.
    Returns a ready‑to‑train :class:`QCNNAutoencoderModel`.
    """
    return QCNNAutoencoderModel(input_dim, latent_dim, hidden_dims)


__all__ = ["QCNNAutoencoder", "QCNNAutoencoderModel"]
