"""Hybrid CNN + QCNN inspired classical network.

This module defines :class:`HybridQCNNNet` which builds on the
convolutional backbone from the original QCNet and then augments it
with a classical QCNN‑style fully‑connected stack before delegating
the final decision to a quantum expectation head implemented in the
QML module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the quantum head from the QML module
from.quantum_qcnn import compute_quantum_expectation


class HybridQCNNNet(nn.Module):
    """Convolutional network followed by a classical QCNN stack and a quantum head."""

    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully‑connected stack mimicking the QCNN feature map
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Classical QCNN‑style layers
        self.qcnn = nn.Sequential(
            nn.Linear(4, 8), nn.Tanh(),
            nn.Linear(8, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh(),
            nn.Linear(4, 4), nn.Tanh(),
            nn.Linear(4, 1)
        )

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

        # Pass through classical QCNN stack
        qcnn_out = self.qcnn(x)

        # Quantum expectation head
        quantum_out = compute_quantum_expectation(qcnn_out)

        # Return class probabilities
        return torch.cat((quantum_out, 1 - quantum_out), dim=-1)


__all__ = ["HybridQCNNNet"]
