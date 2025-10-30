"""Hybrid classical network combining QCNN layers with a sampler‑style classifier.

The network mirrors the original QCNNModel but replaces the final linear head
with a two‑output soft‑max layer inspired by SamplerQNN.  This allows the
model to produce class probabilities while still leveraging the depth and
feature‑map ideas of the QCNN architecture.

The class is fully compatible with PyTorch training pipelines and can be used
directly in a `torch.nn.Module` container.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class QCNNGen012Model(nn.Module):
    """Classical convolution‑inspired network with a sampler head.

    The network consists of a feature map, three convolution‑pooling stages,
    and a soft‑max output layer.  The architecture is a direct extension of
    the original QCNNModel with an additional SamplerQNN‑like head.
    """

    def __init__(self) -> None:
        super().__init__()

        # Feature extraction
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())

        # First convolution‑pooling block
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())

        # Second convolution‑pooling block
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())

        # Third convolution‑pooling block
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.pool3 = nn.Sequential(nn.Linear(4, 2), nn.Tanh())

        # Soft‑max classifier (SamplerQNN style)
        self.head = nn.Linear(2, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass through the feature map, conv‑pool blocks, and head."""
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        logits = self.head(x)
        return F.softmax(logits, dim=-1)


def QCNNGen012() -> QCNNGen012Model:
    """Factory returning a configured :class:`QCNNGen012Model`."""
    return QCNNGen012Model()


__all__ = ["QCNNGen012", "QCNNGen012Model"]
