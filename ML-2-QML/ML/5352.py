"""Hybrid Neural Architecture combining classical CNN, quanvolution, and a simulated quantum fully connected layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """Classical 2×2 quanvolution filter: a 2×2 convolution producing 4 feature maps."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class FCL(nn.Module):
    """Simulated quantum fully‑connected layer – a linear mapping followed by a tanh non‑linearity."""
    def __init__(self, in_features: int, out_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.tanh(self.linear(x))


class HybridNATModel(nn.Module):
    """Hybrid classical‑quantum inspired architecture.
    Combines a shallow CNN, a quanvolution filter, and a simulated quantum fully‑connected layer
    before a final linear classifier. The design preserves the modularity of the original
    Quantum‑NAT and enriches it with data‑driven feature extraction from the quanvolution block.
    """
    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        # Classical convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Quanvolution block
        self.quanv = QuanvolutionFilter()
        # Simulated quantum fully‑connected layer
        # Input dimensionality: conv output (16×7×7=784) + quanv output (4×14×14=784) = 1568
        self.fcl = FCL(in_features=1568)
        # Final classifier
        self.classifier = nn.Linear(1, n_classes)
        self.norm = nn.BatchNorm1d(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Classical convolutional path
        conv_feat = self.features(x)
        conv_flat = conv_feat.view(conv_feat.size(0), -1)

        # Quanvolution path
        quanv_feat = self.quanv(x)

        # Concatenate and pass through simulated quantum layer
        concat = torch.cat([conv_flat, quanv_feat], dim=1)
        q_out = self.fcl(concat)

        # Classification head
        logits = self.classifier(q_out)
        return self.norm(logits)


__all__ = ["HybridNATModel"]
