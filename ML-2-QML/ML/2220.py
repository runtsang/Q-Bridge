"""Unified hybrid classifier: classical core + quantum head."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Classical encoder
# --------------------------------------------------------------------------- #
class ResNetBlock(nn.Module):
    """A lightweight residual block for image feature extraction."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride!= 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

# --------------------------------------------------------------------------- #
# Hybrid head (sigmoid with optional shift)
# --------------------------------------------------------------------------- #
class HybridHead(nn.Module):
    """Dense head that mimics a quantum expectation output."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x).squeeze(-1)
        probs = torch.sigmoid(logits + self.shift)
        return torch.stack([probs, 1 - probs], dim=-1)

# --------------------------------------------------------------------------- #
# Build function
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """
    Construct a hybrid classifier that consists of a convolutional backbone,
    a fully‑connected stack, and a sigmoid head.  The function returns
    metadata that mirrors the quantum construction: an encoding list,
    the number of trainable parameters per layer, and a dummy observable
    list that is compatible with the quantum side.

    Parameters
    ----------
    num_features : int
        Number of output features from the last fully‑connected layer
        (also the dimensionality of the quantum input).
    depth : int
        Depth of the fully‑connected stack.

    Returns
    -------
    network : nn.Module
        The complete PyTorch model.
    encoding : List[int]
        Identity mapping of the classical features to the quantum circuit.
    weight_sizes : List[int]
        Number of trainable parameters per linear layer.
    observables : List[int]
        Dummy observable identifiers (0 and 1) to match the quantum output.
    """
    # Convolutional backbone
    backbone = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        ResNetBlock(16, 32, stride=2),
        ResNetBlock(32, 64, stride=2),
        nn.AdaptiveAvgPool2d((1, 1)),
    )

    # Flatten and fully‑connected stack
    layers: List[nn.Module] = [backbone, nn.Flatten()]
    in_dim = 64
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU(inplace=True))
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    # Hybrid head
    head = HybridHead(in_dim)
    layers.append(head)

    network = nn.Sequential(*layers)
    observables = [0, 1]  # placeholder matching quantum head
    return network, encoding, weight_sizes, observables

__all__ = ["build_classifier_circuit"]
