"""
Quantum version of the hybrid binary classifier.
Uses torchquantum to build a parameterised two‑qubit circuit
with a random layer and rotation gates.  The classical
convolutional backbone feeds the feature vector into the
quantum encoder, and a linear measurement head produces the
output probability.  The interface matches the classical
counterpart for direct benchmark comparisons.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.quantum as tq  # torchquantum
import numpy as np


class QuantumEncoder(tq.QuantumModule):
    """Random rotation layer that expands classical features into a quantum state."""

    def __init__(self, n_wires: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)


class HybridQuantumHead(tq.QuantumModule):
    """Quantum expectation head with a linear post‑processing layer."""

    def __init__(self, n_wires: int = 2):
        super().__init__()
        self.encoder = QuantumEncoder(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.encoder.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


class QuantumHybridClassifier(nn.Module):
    """
    Convolutional backbone followed by a quantum hybrid head.
    The architecture of the CNN matches the classical version to
    ensure that any performance difference stems from the head.
    """

    def __init__(self) -> None:
        super().__init__()
        # Convolutional front‑end (identical to the classical model)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully‑connected backbone
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum hybrid head
        self.head = HybridQuantumHead(n_wires=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.drop2(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.head(x)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuantumEncoder", "HybridQuantumHead", "QuantumHybridClassifier"]
