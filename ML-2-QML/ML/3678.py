"""Hybrid classical convolutional and classifier module."""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple

class ConvFilter(nn.Module):
    """Classical convolutional filter emulating a quantum filter."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply the filter to a 2‑D tensor of shape (kernel_size, kernel_size).

        Returns a 2‑D activation map that can be flattened and fed into a classifier.
        """
        data = data.unsqueeze(0).unsqueeze(0)  # add batch & channel dims
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.squeeze()  # shape (1, kernel_size, kernel_size)

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, list[int], list[int], list[int]]:
    """
    Build a feed‑forward classifier mirroring the quantum helper interface.
    Returns network, encoding indices, weight sizes, and output observables.
    """
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

class ConvGen(nn.Module):
    """
    Combined classical convolutional filter and classifier.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, depth: int = 3) -> None:
        super().__init__()
        self.filter = ConvFilter(kernel_size=kernel_size, threshold=threshold)
        self.classifier, _, _, _ = build_classifier_circuit(num_features=kernel_size**2, depth=depth)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Run the filter and feed the flattened activation map into the classifier.
        Expected input shape: (kernel_size, kernel_size).
        """
        activations = self.filter(data)            # shape (1, kernel_size, kernel_size)
        flat = activations.view(-1)                # shape (kernel_size**2)
        logits = self.classifier(flat)             # shape (2)
        return logits

__all__ = ["ConvGen"]
