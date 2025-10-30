"""Hybrid classical model combining CNN, fully connected layers, and a quantum-inspired expectation layer."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable


class QuantumExpectationLayer(nn.Module):
    """Classical layer that emulates a quantum fully connected layer by computing the expectation
    value of a tanh-activated linear transform. This mimics the behaviour of the quantum
    fully connected layer in the reference pair 2."""
    def __init__(self, in_features: int, out_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute tanh of linear output and average over batch to emulate expectation
        return torch.tanh(self.linear(x)).mean(dim=0, keepdim=True)


class HybridNATModel(nn.Module):
    """Hybrid classical convolutional network that ends with a quantum-inspired expectation layer.

    The architecture mirrors the classical CNN from the original QuantumNAT model while
    replacing the final fully‑connected projection with a QuantumExpectationLayer.
    This makes the model amenable to quantum‑classical hybrid experiments and
    allows easy swapping of the expectation layer for a true quantum circuit.
    """
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)
        self.q_expect = QuantumExpectationLayer(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x = self.features(x)
        x = x.view(bsz, -1)
        x = self.fc(x)
        x = self.norm(x)
        # Pass the 4‑dimensional feature vector through the quantum‑inspired layer
        return self.q_expect(x)


__all__ = ["HybridNATModel"]
