"""Enhanced classical counterpart of the QCNN with richer feature extraction and advanced training hooks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import relu, dropout
from typing import Iterable

__all__ = ["QCNNPlus", "QCNNPlusFactory", "hybrid_loss"]

class QCNNPlus(nn.Module):
    """
    A deeper, residual‑style feed‑forward network that mimics the quantum convolution‑pooling
    hierarchy.  Each block has a **feature‑map** (Linear‑Tanh) and a **convolution‑like
    1‑D convolution (Linear‑Tanh).   The network uses skip connections between the
    feature‑map and the convolutional blocks to preserve information, and optional
    dropout for regularisation.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: Iterable[int] | None = None,
        drop_prob: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 24, 32]

        # Feature map
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh(),
        )

        # Convolution‑like blocks with skip connections
        self.conv1 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.Tanh(),
        )
        self.pool1 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
        )

        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1]),
            nn.Tanh(),
        )
        self.pool2 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.Tanh(),
        )

        self.conv3 = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[2]),
            nn.Tanh(),
        )

        self.head = nn.Linear(hidden_dims[2], 1)

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        # Residual connection: feature_map -> conv1
        x = self.conv1(x) + x
        x = self.dropout(x)
        x = self.pool1(x)

        # Residual connection: pool1 -> conv2
        x = self.conv2(x) + x
        x = self.dropout(x)
        x = self.pool2(x)

        # Residual connection: pool2 -> conv3
        x = self.conv3(x) + x
        x = self.dropout(x)

        out = self.head(x)
        return torch.sigmoid(out)

def QCNNPlusFactory(
    input_dim: int = 8,
    hidden_dims: Iterable[int] | None = None,
    drop_prob: float = 0.1,
) -> QCNNPlus:
    """
    Factory returning a configured :class:`QCNNPlus` instance.
    """
    return QCNNPlus(input_dim=input_dim, hidden_dims=hidden_dims, drop_prob=drop_prob)

def hybrid_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantum_pred: torch.Tensor,
    weight: float = 0.5,
) -> torch.Tensor:
    """
    Compute a hybrid loss that blends standard MSE with a quantum‑fidelity‑like term.
    The quantum term is a simple proxy that rewards the mean of the quantum prediction
    being close to 1.  In a real workflow the quantum_pred would be a state‑vector or
    expectation value obtained from a quantum circuit.

    Parameters
    ----------
    pred : torch.Tensor
        Classical prediction.
    target : torch.Tensor
        Ground‑truth labels.
    quantum_pred : torch.Tensor
        Quantum prediction (e.g. expectation value in [0,1]).
    weight : float
        Weight of the quantum term in the total loss.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    mse = nn.functional.mse_loss(pred, target)
    # Fidelity‑like term: (1 - mean(quantum_pred))  -> lower is better
    fidelity_term = 1.0 - quantum_pred.mean()
    return mse + weight * fidelity_term
