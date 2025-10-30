"""QCNNModel: a classical convolution‑inspired network with residuals, batch‑norm, and dropout."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class QCNNModel(nn.Module):
    """
    Classical QCNN‑inspired architecture.

    Architecture:
        * Feature map: Linear(8→32) → BatchNorm → ReLU
        * Conv‑1: Linear(32→32) → Dropout → BatchNorm → ReLU
        * Pool‑1: Linear(32→24) → BatchNorm → ReLU
        * Conv‑2: Linear(24→24) → Dropout → BatchNorm → ReLU
        * Pool‑2: Linear(24→12) → BatchNorm → ReLU
        * Conv‑3: Linear(12→12) → Dropout → BatchNorm → ReLU
        * Pool‑3: Linear(12→6) → BatchNorm → ReLU
        * Head: Linear(6→1) → Sigmoid
    Residual connections are added from the feature map to conv‑2 and from conv‑1 to pool‑1.
    """

    def __init__(self, in_features: int = 8, seed: int | None = None) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.feature_map = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Linear(32, 32),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.Sequential(
            nn.Linear(32, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(24, 24),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.Sequential(
            nn.Linear(24, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Linear(12, 12),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(12),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.Sequential(
            nn.Linear(12, 6),
            nn.BatchNorm1d(6),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(6, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature map
        x_f = self.feature_map(x)

        # Conv‑1
        x = self.conv1(x_f)

        # Residual connection from feature map to conv‑2
        x_res1 = x_f
        x = self.pool1(x)

        # Conv‑2 with residual
        x = self.conv2(x + x_res1)

        x = self.pool2(x)

        # Conv‑3
        x = self.conv3(x)

        x = self.pool3(x)

        logits = self.head(x)
        probs = torch.sigmoid(logits)
        return probs

    def freeze_feature_map(self) -> None:
        """Freeze the feature‑map layers so that only the remaining layers are trainable."""
        for param in self.feature_map.parameters():
            param.requires_grad = False

    def unfreeze_feature_map(self) -> None:
        """Unfreeze the feature‑map layers."""
        for param in self.feature_map.parameters():
            param.requires_grad = True


def QCNN() -> QCNNModel:
    """Convenience factory returning an initialized QCNNModel."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
