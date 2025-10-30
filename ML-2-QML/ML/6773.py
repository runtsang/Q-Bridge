"""Classical hybrid neural network for binary classification with optional quantum-inspired head.

This module defines a CNN feature extractor followed by a hybrid dense head that can
either apply a differentiable sigmoid (classical) or emulate a quantum expectation
through a parameterized linear layer.  It also includes a lightweight sampler head
implemented as a small feed‑forward network, mirroring the SamplerQNN interface
from the quantum side.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalHybridHead(nn.Module):
    """Classical dense head that emulates a quantum expectation via a linear layer
    followed by a sigmoid.  The shift parameter allows calibration similar to the
    quantum shift used in the original hybrid model.
    """
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x).squeeze(-1)
        probs = torch.sigmoid(logits + self.shift)
        return torch.cat([probs, 1 - probs], dim=-1)


class ClassicalSamplerHead(nn.Module):
    """A classical approximation of a quantum sampler head.
    The network outputs a two‑dimensional probability vector via a softmax.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


class ClassicalHybridNet(nn.Module):
    """CNN feature extractor followed by a choice of head:
    * 'dense'  – ClassicalHybridHead
    *'sampler' – ClassicalSamplerHead
    """
    def __init__(self, head: str = "dense") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

        if head == "dense":
            self.head = ClassicalHybridHead(self.fc3.out_features)
        elif head == "sampler":
            self.head = ClassicalSamplerHead()
        else:
            raise ValueError(f"Unsupported head type {head!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        probs = self.head(logits)
        return probs


__all__ = ["ClassicalHybridHead", "ClassicalSamplerHead", "ClassicalHybridNet"]
