"""Hybrid classical QCNN implementation with fully‑connected refinement.

This class combines a classical convolution‑inspired feature extractor
with a fully connected refinement stage inspired by the FCL example.
It is designed to be interchangeable with the quantum version for
benchmarking hybrid versus purely classical performance.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np

class QCNNGen351(nn.Module):
    """
    Classical QCNN with fully‑connected refinement.

    The network consists of:
        * A feature map converting 8‑dimensional input into a 16‑dimensional
          latent representation.
        * Three convolution‑pooling blocks mirroring the quantum construction.
        * A fully connected layer that aggregates the final features into a
          scalar prediction.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 16) -> None:
        super().__init__()
        # Feature extractor
        self.feature_map = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())

        # Convolution‑pooling blocks
        self.conv1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(hidden_dim, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

        # Fully connected refinement (inspired by FCL)
        self.fcl = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the QCNN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 8).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, 1) with values in (0, 1).
        """
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        # Fully connected output
        out = torch.tanh(self.fcl(x)).mean(dim=0, keepdim=True)
        return torch.sigmoid(out)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper for NumPy inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (n_samples, 8).

        Returns
        -------
        np.ndarray
            Array of predicted probabilities.
        """
        self.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(inputs, dtype=torch.float32)
            pred = self.forward(tensor).cpu().numpy()
        return pred.squeeze()

def QCNNGen351Model() -> QCNNGen351:
    """Factory returning the configured :class:`QCNNGen351` model."""
    return QCNNGen351()

__all__ = ["QCNNGen351", "QCNNGen351Model"]
