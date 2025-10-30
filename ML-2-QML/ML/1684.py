"""
`QCNNModel` – a residual, regularised classical analogue of a QCNN.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class QCNNModel(nn.Module):
    """
    A classical neural network that mimics the structure of a QCNN.
    The network uses residual connections, batch normalisation, and dropout
    to improve generalisation while preserving the original module
    signature.

    Parameters
    ----------
    in_features : int, optional
        Dimensionality of the input vector (default 8).
    hidden_features : list[int], optional
        List of hidden layer sizes.  Each entry defines a convolutional
        stage.  The default matches the original 16‑16‑12‑8‑4‑4 diagram.
    dropout : float, optional
        Dropout probability applied after each stage (default 0.1).
    """

    def __init__(
        self,
        in_features: int = 8,
        hidden_features: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_features is None:
            hidden_features = [16, 16, 12, 8, 4, 4]

        self.feature_map = nn.Sequential(
            nn.Linear(in_features, hidden_features[0]),
            nn.Tanh(),
        )
        self.stages = nn.ModuleList()
        prev = hidden_features[0]
        for idx, h in enumerate(hidden_features[1:]):
            stage = nn.Sequential(
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.Tanh(),
                nn.Dropout(dropout),
            )
            # Residual connection if dimensions match
            if prev == h:
                stage = nn.Sequential(
                    stage,
                    nn.Identity(),
                )
            self.stages.append(stage)
            prev = h

        self.head = nn.Linear(hidden_features[-1], 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        for stage in self.stages:
            residual = x
            x = stage(x)
            # Add residual only if shapes match
            if residual.shape == x.shape:
                x = x + residual
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNModel:
    """Factory for the classical QCNN‑style network."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
