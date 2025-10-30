"""Enhanced classical quanvolution module with deeper feature extractor and multi-output support."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionEnhanced(nn.Module):
    """
    Classical quanvolution model that emulates the quantum filter using a deeper convolutional
    pipeline and supports multi‑output classification or regression.
    """

    def __init__(self, num_outputs: int = 10, num_classes: bool = True) -> None:
        """
        Args:
            num_outputs: Number of output units (default 10 for MNIST classification).
            num_classes: If True, the final activation will be log_softmax for classification.
                         If False, raw logits are returned for regression or other tasks.
        """
        super().__init__()
        self.num_outputs = num_outputs
        self.num_classes = num_classes

        # Deeper convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        # Flatten and linear head
        self.classifier = nn.Linear(32 * 7 * 7, self.num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classical quanvolution network.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            Log‑softmax logits if num_classes is True, otherwise raw logits.
        """
        feats = self.features(x)
        flat = feats.view(feats.size(0), -1)
        logits = self.classifier(flat)
        if self.num_classes:
            return F.log_softmax(logits, dim=-1)
        return logits


__all__ = ["QuanvolutionEnhanced"]
