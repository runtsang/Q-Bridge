"""Hybrid classical model combining a CNN backbone with a quantum‑inspired fully connected layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridQuantumNATModel(nn.Module):
    """Classical CNN + quantum‑inspired fully connected layer.

    The network first extracts spatial features with a small CNN.  The flattened
    representation is then passed through a two‑layer MLP that emulates the
    quantum fully‑connected block from the original Quantum‑NAT paper:
    a linear map followed by a non‑linear activation (Tanh) and a final
    projection to four output features.  Batch‑norm is applied to stabilize
    training.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.size(0)
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)


__all__ = ["HybridQuantumNATModel"]
