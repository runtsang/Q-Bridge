"""Hybrid self‑attention + QCNN model – classical implementation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Classical self‑attention helper
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """
    Implements a simple dot‑product self‑attention block.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the feature space.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable linear maps for query, key and value
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the self‑attention block.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Attention‑weighted representation of shape (batch, seq_len, embed_dim).
        """
        q = self.q_linear(inputs)
        k = self.k_linear(inputs)
        v = self.v_linear(inputs)

        scores = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1
        )
        return torch.matmul(scores, v)


# --------------------------------------------------------------------------- #
# QCNN inspired fully‑connected stack
# --------------------------------------------------------------------------- #
class QCNNModel(nn.Module):
    """
    A lightweight QCNN‑style network that emulates the quantum convolution and
    pooling steps with fully‑connected layers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


# --------------------------------------------------------------------------- #
# Hybrid model that stitches the two components together
# --------------------------------------------------------------------------- #
class HybridSelfAttentionQCNN(nn.Module):
    """
    Hybrid classical model that first applies self‑attention, then feeds
    the result through a QCNN‑style network.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the self‑attention feature space.
    """

    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim)
        self.qcnn = QCNNModel()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid model.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output of the QCNN head, shape (batch, 1).
        """
        attn_out = self.attention(inputs)
        # Flatten the attention output before feeding into QCNN
        flattened = attn_out.view(attn_out.size(0), -1)
        return self.qcnn(flattened)


def HybridSelfAttentionQCNNFactory(embed_dim: int = 4) -> HybridSelfAttentionQCNN:
    """
    Factory function that returns a ready‑to‑use hybrid model.

    Parameters
    ----------
    embed_dim : int, optional
        Dimensionality of the self‑attention feature space (default 4).

    Returns
    -------
    HybridSelfAttentionQCNN
    """
    return HybridSelfAttentionQCNN(embed_dim)


__all__ = ["HybridSelfAttentionQCNN", "HybridSelfAttentionQCNNFactory"]
