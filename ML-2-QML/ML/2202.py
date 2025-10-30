"""Hybrid Quanvolution model – purely classical implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionHybridML(nn.Module):
    """
    Classical network that mimics the structure of a quanvolutional filter
    followed by a fully‑connected classifier.  It combines two convolutional
    stages with batch‑normalisation, dropout and a final linear head.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor – two conv stages with BN and dropout
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(2),
        )
        # Fully‑connected head – two linear layers with ReLU
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.features(x)
        flattened = features.view(x.size(0), -1)
        logits = self.classifier(flattened)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybridML"]
