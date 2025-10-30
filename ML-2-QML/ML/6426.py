"""Hybrid classical model combining CNN, fully connected layers, and a parameterized
fully connected layer inspired by the FCL example.

The model extends the original QuantumNAT CNN but replaces the final linear layer
with a small trainable “FCL” block that computes the expectation of a tanh
transformed linear output.  This demonstrates how a purely classical network
can mimic a quantum‑style measurement while retaining full PyTorch
back‑propagation support.

The architecture is:
    Conv2d -> ReLU -> MaxPool2d
    Conv2d -> ReLU -> MaxPool2d
    Flatten
    Linear(64) -> ReLU
    Linear(32) -> ReLU
    FCL (linear + tanh + mean)
    Linear(4)
    BatchNorm1d
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FCL(nn.Module):
    """A lightweight stand‑in for a quantum fully‑connected layer.

    The original example used a parameterised circuit that evaluated a
    single expectation value.  Here we emulate that behaviour with a
    linear transform followed by a tanh activation and a mean across
    the feature axis.  The module is trainable and fully differentiable.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x`` is expected to be of shape (*, in_features)
        out = self.linear(x)              # (*, 1)
        out = torch.tanh(out)             # (*, 1)
        # Reduce to a single scalar per sample
        return out.mean(dim=-1, keepdim=True)  # (*, 1)


class QFCModel(nn.Module):
    """Hybrid classical model inspired by the Quantum‑NAT paper.

    Combines a standard convolutional front‑end with a small
    fully‑connected backbone and a quantum‑style FCL block.  The
    final output is normalised with a 1‑D batch‑norm layer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fcl = _FCL(32)
        self.fc3 = nn.Linear(1, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = F.relu(self.fc1(flattened))
        out = F.relu(self.fc2(out))
        out = self.fcl(out)          # (*, 1)
        out = self.fc3(out)          # (*, 4)
        return self.norm(out)


__all__ = ["QFCModel"]
