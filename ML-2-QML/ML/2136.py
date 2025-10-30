"""Enhanced classical QCNN with dropout and batch‑norm.

The architecture mirrors the original fully‑connected stack but introduces
dropout layers, batch‑normalisation and a small residual skip connection
to improve generalisation and training stability.  The module keeps the
public API identical to the seed – a factory ``QCNN()`` that returns a
``QCNNModel`` instance ready for use with ``torch``.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple


class QCNNModel(nn.Module):
    """Classical QCNN with dropout, batch‑norm and a residual connection."""

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

        # Feature map
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.Dropout(0.2),
        )

        # Conv / Pool blocks
        self.block1 = self._make_block(16, 16, name="conv1")
        self.pool1 = self._make_block(16, 12, name="pool1")
        self.block2 = self._make_block(12, 8, name="conv2")
        self.pool2 = self._make_block(8, 4, name="pool2")
        self.block3 = self._make_block(4, 4, name="conv3")

        # Residual projector (linear) to match dimensions
        self.res_proj = nn.Linear(16, 4)

        # Head
        self.head = nn.Linear(4, 1)

        self._init_weights()

    def _make_block(self, in_features: int, out_features: int, *, name: str) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.LayerNorm(out_features, elementwise_affine=False),
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x1 = self.block1(x)
        x2 = self.pool1(x1)
        x3 = self.block2(x2)
        x4 = self.pool2(x3)
        x5 = self.block3(x4)

        # Residual: project feature map to match final dimension and add
        res = self.res_proj(x)
        out = torch.sigmoid(self.head(x5 + res))

        return out

    def get_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the activations before the final head."""
        x = self.feature_map(inputs)
        x1 = self.block1(x)
        x2 = self.pool1(x1)
        x3 = self.block2(x2)
        x4 = self.pool2(x3)
        x5 = self.block3(x4)
        return x5


def QCNN() -> QCNNModel:
    """Factory producing the enhanced classical QCNN."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
