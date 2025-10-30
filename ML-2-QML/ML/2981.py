"""Hybrid classical classifier with quantum-inspired components."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List

class QuantumFeatureEncoder(nn.Module):
    """Encodes classical features using a sinusoidal mapping mimicking quantum rotations."""
    def __init__(self, input_dim: int, encoding_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.linear = nn.Linear(input_dim, encoding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply a linear map followed by sin/cos to emulate Rx/RY rotations
        z = self.linear(x)
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)

class QuanvolutionFilter(nn.Module):
    """Classical 2x2 convolution followed by a quantum kernel approximation."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class HybridClassifier(nn.Module):
    """Hybrid network: quanvolution filter + quantum-inspired encoder + linear head."""
    def __init__(self, num_classes: int = 10, depth: int = 3):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # Assume output of filter is 4*14*14 features
        feat_dim = 4 * 14 * 14
        self.encoder = QuantumFeatureEncoder(feat_dim, feat_dim // 2)
        self.head = nn.Linear(self.encoder.encoding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        encoded = self.encoder(features)
        logits = self.head(encoded)
        return F.log_softmax(logits, dim=-1)

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Sequential, Iterable[int], Iterable[int], List[int]]:
    """Construct a classical feed‑forward classifier mirroring the quantum interface."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        lin = nn.Linear(in_dim, num_features)
        layers.extend([lin, nn.ReLU()])
        weight_sizes.append(lin.weight.numel() + lin.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    net = nn.Sequential(*layers)
    observables = list(range(2))
    return net, encoding, weight_sizes, observables

__all__ = ["HybridClassifier", "build_classifier_circuit"]
