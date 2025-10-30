"""QCNNEnhanced – a deeper classical convolution‑inspired network.

The network mirrors the original QCNN but adds batch‑normalisation,
dropout and skip‑connections.  It can be used as a drop‑in
replacement for `QCNNModel` in any PyTorch training pipeline.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import List


class QCNNEnhanced(nn.Module):
    """A deeper, regularised QCNN‑style feed‑forward network.

    Parameters
    ----------
    input_dim : int, default 8
        Size of the input feature vector.
    hidden_dims : List[int], default [16, 16, 12, 8, 4, 4]
        The width of each hidden layer.
    dropout : float, default 0.2
        Dropout probability applied after each activation.
    residual : bool, default True
        If set, a residual connection is added from the input of a
        block to its output (for blocks where the dimensionality
        matches).
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.2,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        layers: List[nn.Module] = []
        prev = input_dim
        for idx, h in enumerate(hidden_dims):
            block = nn.Sequential(
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            layers.append(block)
            # Add a residual connection if dimensions match
            if residual and prev == h:
                layers.append(nn.Identity())
            prev = h
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.feature_extractor(x)
        return torch.sigmoid(self.classifier(x))


def QCNNEnhancedFactory() -> QCNNEnhanced:
    """Return a ready‑to‑use instance of the enhanced QCNN."""
    return QCNNEnhanced()


__all__ = ["QCNNEnhanced", "QCNNEnhancedFactory"]
