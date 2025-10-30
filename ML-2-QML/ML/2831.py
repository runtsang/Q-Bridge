"""Python module implementing a hybrid classical‑quantum filter and classifier.

The classical branch learns a 2×2 convolution with a sigmoid gating
controlled by a threshold.  The quantum branch is supplied as a
callable that returns a flat feature vector.  Both branches are
concatenated and fed to a linear head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional


class ClassicalFilter(nn.Module):
    """Learnable 2×2 convolution with a sigmoid gating.

    The filter is fully differentiable and can be trained jointly
    with the rest of the network.  The `threshold` parameter
    determines the bias of the sigmoid gate.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)
        gated = torch.sigmoid(conv_out - self.threshold)
        return gated


class HybridQuanvolution(nn.Module):
    """Drop‑in replacement that concatenates classical and quantum features.

    Parameters
    ----------
    quantum_feature_fn : Callable[[torch.Tensor], torch.Tensor] | None, optional
        A callable that accepts a batch of images and returns a flat
        quantum feature vector.  If ``None`` the quantum branch is
        omitted.
    num_classes : int, default 10
        Number of output classes for the classifier head.
    """
    def __init__(
        self,
        quantum_feature_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.classical = ClassicalFilter()
        self.quantum = quantum_feature_fn

        # Classical filter outputs (B, 1, 14, 14) → 196 features.
        # Quantum filter (if present) outputs 4 × 14 × 14 → 784 features.
        total_features = 196 + (784 if self.quantum is not None else 0)
        self.classifier = nn.Linear(total_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        class_feat = self.classical(x)          # (B, 1, 14, 14)
        class_flat = class_feat.view(x.size(0), -1)  # (B, 196)

        if self.quantum is not None:
            quantum_flat = self.quantum(x)     # (B, 784)
            features = torch.cat([class_flat, quantum_flat], dim=1)
        else:
            features = class_flat

        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["ClassicalFilter", "HybridQuanvolution"]
