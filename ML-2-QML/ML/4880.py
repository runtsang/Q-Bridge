"""Hybrid layer that fuses classical feature extraction with a simple feed‑forward network.

The architecture is a shallow network inspired by QCNN and EstimatorQNN, ending
in a softmax over two classes.  It is fully compatible with PyTorch and
does not depend on any quantum libraries.
"""

from __future__ import annotations

import torch
from torch import nn

class UnifiedHybridLayer(nn.Module):
    """Classical hybrid layer – a lightweight neural network."""
    def __init__(self, in_features: int = 2, hidden: int = 8, num_classes: int = 2) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(hidden // 2, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)
        out = self.classifier(feat)
        return self.softmax(out)
