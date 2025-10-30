"""Hybrid quantum‑inspired neural network combining a classical convolutional filter
and a parameterized fully connected quantum layer.

The class exposes a ``run`` method that accepts a list of parameters and returns
a single expectation value, mimicking the behaviour of the original FCL example,
while the ``forward`` method implements the classical convolution‑plus‑linear
pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable

class HybridQuantumLayer(nn.Module):
    """
    A hybrid neural network that merges a classical convolutional filter (inspired
    by the quanvolution example) with a parameterized fully connected quantum
    layer (inspired by the FCL example).

    The ``forward`` method implements the classical part, while the ``run`` method
    evaluates the quantum‑inspired expectation value from a list of parameters.
    """

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        # Classical convolutional filter (2x2 kernel, stride 2)
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Linear head matching the output of the conv filter (28x28 -> 14x14 patches)
        self.linear = nn.Linear(4 * 14 * 14, 10)
        # Quantum‑inspired fully connected layer
        self.q_linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Classical forward pass: applies the convolutional filter, flattens the
        feature map, and produces log‑softmax logits for a 10‑class classification.
        """
        features = self.conv(x)
        features = features.view(x.size(0), -1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Quantum‑inspired evaluation: given a list of parameters ``thetas``, the method
        feeds them into a linear layer, applies a tanh non‑linearity, and returns the
        mean expectation value over the batch.  The output is a 1‑D tensor of shape
        (1,).
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.q_linear(values)).mean(dim=0)
        return expectation.detach()

__all__ = ["HybridQuantumLayer"]
