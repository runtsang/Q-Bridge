"""Hybrid classical convolution + classifier.

This module provides a drop‑in replacement for the original Conv.py
while adding a fully‑connected classifier on top of the learned
features.  The interface mirrors the quantum implementation so that
experiments can be run on either backend with minimal changes.

Usage
-----
>>> from Conv__gen134 import HybridConvClassifier
>>> model = HybridConvClassifier(kernel_size=3, threshold=0.5, classifier_depth=3)
>>> logits = model.run([[1,0,1],[0,1,0],[1,1,1]])  # 3×3 sample
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Tuple

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Construct a feed‑forward classifier and metadata similar to the quantum variant."""
    layers: list[nn.Module] = []
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

class HybridConvClassifier(nn.Module):
    """
    Classical convolutional filter followed by a fully‑connected classifier.
    The class is intentionally lightweight so that it can be swapped for
    the quantum implementation without touching the surrounding code.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, classifier_depth: int = 2) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold

        # --- Convolutional feature extractor ---------------------------------
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # --- Classifier ------------------------------------------------------
        num_features = kernel_size ** 2
        self.classifier, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features=num_features, depth=classifier_depth
        )

    def forward(self, data: torch.Tensor | Iterable[Iterable[Iterable[float]]]) -> torch.Tensor:
        """
        Forward pass that accepts either a torch tensor or a nested list.
        The input is expected to be a single 2‑D sample or a batch of such samples.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)

        # Ensure shape (batch, 1, H, W)
        if data.dim() == 2:
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.dim() == 3:
            data = data.unsqueeze(1)
        elif data.dim()!= 4:
            raise ValueError(f"Unsupported input shape {data.shape}")

        # Convolution and activation
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)

        # Flatten and classify
        flat = activations.view(activations.size(0), -1)
        logits = self.classifier(flat)
        return logits

    # Compatibility shim ----------------------------------------------------
    def run(self, data: torch.Tensor | Iterable[Iterable[Iterable[float]]]) -> torch.Tensor:
        """Alias for forward to match the original Conv.run signature."""
        return self.forward(data)

def Conv() -> HybridConvClassifier:
    """
    Factory that returns a ready‑to‑use HybridConvClassifier.
    Mirrors the original Conv() API so existing code continues to work.
    """
    return HybridConvClassifier()

__all__ = ["HybridConvClassifier", "Conv"]
