"""
FraudDetectionHybrid: a dual classical–quantum fraud detection model.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from dataclasses import dataclass
from typing import Iterable, Tuple

# ------------------------------------------------------------------
#  Classical sub‑module
# ------------------------------------------------------------------
class AttentionBlock(nn.Module):
    """Self‑attention layer with trainable rotation and entanglement
    parameters.  The parameters are reshaped to match the
    4‑dimensional embedding used in the original SelfAttention helper.
    """
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # rotation_params: (embed_dim, embed_dim)
        self.rotation_params = Parameter(torch.randn(embed_dim, embed_dim))
        # entangle_params: (embed_dim, embed_dim)
        self.entangle_params = Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features)
        query = x @ self.rotation_params
        key   = x @ self.entangle_params
        scores = torch.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        return scores @ x


class QCNNModel(nn.Module):
    """Lightweight QCNN‑style MLP emulating convolution, pooling and
    final classification.  Inspired by the classical QCNN helper.
    """
    def __init__(self, input_dim: int = 8) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head   = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


# ------------------------------------------------------------------
#  Hybrid wrapper
# ------------------------------------------------------------------
class FraudDetectionHybrid(nn.Module):
    """
    Combines a classical attention‑augmented QCNN with a quantum QCNN.
    The classical forward path can be used for rapid inference, while
    the quantum path provides an alternative evaluation that may
    capture non‑classical correlations.
    """
    def __init__(self, input_dim: int = 8) -> None:
        super().__init__()
        self.attention = AttentionBlock()
        self.cqcnn     = QCNNModel(input_dim=input_dim)
        # The final linear layer maps the QCNN output to a scalar.
        self.final = nn.Linear(1, 1)

    def forward_classical(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classical forward pass: attention → QCNN → sigmoid.
        """
        attn_out = self.attention(x)
        qcnn_out = self.cqcnn(attn_out)
        return torch.sigmoid(self.final(qcnn_out))

    def forward_quantum(self, x: torch.Tensor, quantum_module) -> torch.Tensor:
        """
        Quantum forward pass: delegate to a pre‑built quantum module
        that implements the QCNN variational circuit.
        """
        # quantum_module must expose a `predict` method returning a
        # probability tensor of shape (batch, 1).
        return quantum_module.predict(x)


__all__ = ["FraudDetectionHybrid", "AttentionBlock", "QCNNModel"]
