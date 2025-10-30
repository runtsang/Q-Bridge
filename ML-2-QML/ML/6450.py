"""Hybrid classical QCNN + autoencoder model.

This module defines :class:`QCNNAutoEncoder`, a neural network that first reduces
the input dimensionality with a fully‑connected autoencoder and then classifies
the resulting latent vector using a QCNN‑style architecture.  The design mirrors
the two seed projects: the classical QCNN model (``QCNNModel``) and the
autoencoder (``AutoencoderNet``).  The hybrid network is ready for end‑to‑end
training on any PyTorch workflow.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple

# Import seed components
from QCNN import QCNNModel
from Autoencoder import AutoencoderConfig, AutoencoderNet


class QCNNAutoEncoder(nn.Module):
    """Hybrid QCNN + autoencoder classifier.

    The model proceeds in three stages:
    1. **Autoencoder** – encodes the raw input to a latent vector.
    2. **Projection** – maps the latent vector to the 4‑dimensional
       representation expected by the QCNN classifier.
    3. **QCNN** – a stack of fully‑connected layers that mimics the
       quantum convolutional network.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Autoencoder configuration
        self.autoencoder = AutoencoderNet(
            AutoencoderConfig(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        )
        # Linear projection to match QCNN input size (4)
        self.latent_to_qcnn = nn.Linear(latent_dim, 4)
        # QCNN classifier
        self.qcnn = QCNNModel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode, project and classify
        z = self.autoencoder.encode(x)
        z_proj = torch.relu(self.latent_to_qcnn(z))
        logits = self.qcnn(z_proj)
        return logits


def QCNNAutoEncoderFactory(
    input_dim: int,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> QCNNAutoEncoder:
    """Convenience factory mirroring the original seed API."""
    return QCNNAutoEncoder(input_dim, latent_dim, hidden_dims, dropout)


__all__ = ["QCNNAutoEncoder", "QCNNAutoEncoderFactory"]
