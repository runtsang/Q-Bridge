"""Hybrid classical classifier with a quanvolutional feature extractor and a feed‑forward head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilterClassic(nn.Module):
    """Classical 2×2 patch convolution inspired by the quantum filter."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.conv(x).view(x.size(0), -1)


class ClassicalClassifier(nn.Module):
    """Feed‑forward classifier built on top of the classical quanvolution filter."""
    def __init__(self, num_features: int = 4 * 14 * 14, num_classes: int = 10) -> None:
        super().__init__()
        self.feature_extractor = QuanvolutionFilterClassic()
        self.head = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.feature_extractor(x)
        logits = self.head(features)
        return F.log_softmax(logits, dim=-1)


def build_classifier_circuit(num_features: int, depth: int) -> tuple[nn.Module, list[int], list[int], list[int]]:
    """
    Build a classical feed‑forward network that mirrors the quantum ansatz.
    Returns the network, a list of layer sizes, the weight counts per layer, and dummy observables.
    """
    layers = []
    in_dim = num_features
    weight_sizes = []
    for _ in range(depth):
        lin = nn.Linear(in_dim, num_features)
        layers.append(lin)
        layers.append(nn.ReLU())
        weight_sizes.append(lin.weight.numel() + lin.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    net = nn.Sequential(*layers)
    observables = [0] * 2  # placeholder
    return net, list(range(num_features)), weight_sizes, observables


__all__ = ["QuanvolutionFilterClassic", "ClassicalClassifier", "build_classifier_circuit"]
