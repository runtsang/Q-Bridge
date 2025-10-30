"""Hybrid binary classifier – classical implementation.

The module offers three public classes:
* :class:`HybridBinaryClassifier` – CNN + optional quantum head.
* :class:`QCNNModel` – a lightweight fully‑connected QCNN baseline.
* :class:`HybridFunction` – a small differentiable sigmoid head.

The design deliberately mirrors the structure of the original
``ClassicalQuantumBinaryClassification`` seed while adding
QCNN‑style feature mapping and an optional quantum interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridFunction(nn.Module):
    """Purely classical sigmoid head with an optional shift."""

    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x + self.shift)


class QCNNModel(nn.Module):
    """Classical fully‑connected network that mimics the QCNN layers."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


class HybridBinaryClassifier(nn.Module):
    """CNN backbone with an optional quantum hybrid head."""

    def __init__(
        self,
        use_quantum: bool = False,
        quantum_head: nn.Module | None = None,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        if use_quantum:
            if quantum_head is None:
                raise ValueError("quantum_head must be provided when use_quantum=True")
            self.head = quantum_head
        else:
            self.head = HybridFunction(shift)

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
        x = self.fc3(x)
        logits = self.head(x)
        # Convert logits to a two‑class probability distribution
        return torch.cat((logits, 1 - logits), dim=-1)


__all__ = ["HybridFunction", "QCNNModel", "HybridBinaryClassifier"]
