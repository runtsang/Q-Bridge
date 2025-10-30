"""Hybrid classical classifier combining a convolutional feature extractor and a deep feed‑forward head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """Simple 2×2 patch extractor with stride‑2 convolution."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


class HybridQuantumClassifierML(nn.Module):
    """Classical hybrid classifier: classical quanvolution + deep feed‑forward."""
    def __init__(self, num_classes: int = 10, depth: int = 3) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # For 28×28 input the flattened size is 4 * 14 * 14
        in_features = 4 * 14 * 14
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(in_features, in_features))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.head(features)
        return F.log_softmax(logits, dim=-1)


def build_classifier_circuit(num_features: int, depth: int) -> tuple[nn.Module, list[int], list[int], list[int]]:
    """
    Return a classical network and dummy metadata to mirror the quantum API.
    The metadata (encoding, weight_sizes, observables) are placeholders that
    satisfy the interface expected by the quantum side.
    """
    model = HybridQuantumClassifierML(num_classes=2, depth=depth)
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in model.parameters()]
    observables = list(range(2))
    return model, encoding, weight_sizes, observables


__all__ = ["HybridQuantumClassifierML", "build_classifier_circuit"]
