"""Hybrid QCNN – classical implementation.

This module defines :class:`QCNNGen574`, a PyTorch neural network that mirrors the
quantum QCNN architecture but adds a self‑attention layer after each pooling
step.  The design is inspired by the original QCNN and SelfAttention seeds and
demonstrates how classical attention can complement convolution‑pooling
hierarchies.

The network is fully differentiable and can be trained with standard optimisers.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import softmax, sigmoid


class QCNNGen574(nn.Module):
    """
    Classical QCNN with embedded self‑attention.

    Architecture
    ------------
    feature_map -> conv1 -> pool1 -> attn1 -> conv2 -> pool2 -> attn2
    -> conv3 -> pool3 -> attn3 -> head
    """

    def __init__(self, embed_dim: int = 4, num_features: int = 8) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(num_features, 16), nn.Tanh()
        )
        # Convolutional blocks
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Pooling blocks
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.pool3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Self‑attention parameters
        self.attn_weights = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.attn_bias = nn.Parameter(torch.zeros(embed_dim))
        # Final head
        self.head = nn.Linear(4, 1)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute self‑attention over the feature vector.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, embed_dim)

        Returns
        -------
        torch.Tensor
            Attended representation of shape (batch, embed_dim)
        """
        query = torch.matmul(x, self.attn_weights)
        key = torch.matmul(x, self.attn_weights.t())
        scores = softmax(query @ key.t() / (self.attn_weights.shape[0] ** 0.5), dim=-1)
        return scores @ x + self.attn_bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self._attention(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self._attention(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self._attention(x)
        return sigmoid(self.head(x))


def QCNN() -> QCNNGen574:
    """
    Factory that returns a fresh instance of :class:`QCNNGen574`.
    """
    return QCNNGen574()


__all__ = ["QCNN", "QCNNGen574"]
