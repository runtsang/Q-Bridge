"""Hybrid classical classifier with CNN feature extractor and linear head.

This module mirrors the interface of the quantum helper but implements a purely classical
workflow.  It is intended as a drop‑in replacement for the legacy `build_classifier_circuit`
when a quantum backend is unavailable.

The architecture combines a shallow convolutional backbone (inspired by the QFCModel
seeds) with a fully‑connected head that outputs binary logits.  The design is
compatible with the standard PyTorch training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridQuantumClassifier(nn.Module):
    """CNN + linear head for binary classification.

    The network consists of two convolutional blocks followed by a
    two‑layer fully‑connected head.  The feature extractor is identical to
    the classical part of the QFCModel seed, ensuring that the quantum
    variant can reuse the same preprocessing pipeline.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # After two 2×2 poolings on 28×28 input we obtain 7×7 feature maps.
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, height, width).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes).
        """
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        logits = self.fc(flat)
        return self.norm(logits)


__all__ = ["HybridQuantumClassifier"]
