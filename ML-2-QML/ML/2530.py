"""Classical components for the hybrid classifier.

This module defines a lightweight feed‑forward backbone, an optional
convolutional backbone, and a top‑level classifier that delegates the
final decision to a quantum hybrid layer.  The design mirrors the
structure of the original seed while adding support for batched
inputs and a parameter‑efficient quantum head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, List

# Import the quantum components
from quantum_classifier_qml import build_classifier_circuit, Hybrid, QuantumCircuit


class FeedForwardClassifier(nn.Module):
    """Lightweight feed‑forward network that mimics the quantum depth."""
    def __init__(self, num_features: int, depth: int = 2, hidden_dim: int = 32):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = self.output_layer(x)
        return x


class QCNet(nn.Module):
    """Convolutional backbone that mirrors the structure of the original QCNet."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

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
        return x


class QuantumClassifierModel(nn.Module):
    """Hybrid classifier that combines a classical backbone with a quantum expectation head."""
    def __init__(self, num_features: int, depth: int = 2, hidden_dim: int = 32, use_cnn: bool = False):
        super().__init__()
        self.backbone = QCNet() if use_cnn else FeedForwardClassifier(num_features, depth, hidden_dim)
        # Build a quantum circuit that matches the output dimension of the backbone
        self.circuit, _, _, _ = build_classifier_circuit(num_qubits=hidden_dim, depth=depth)
        self.hybrid = Hybrid(n_qubits=hidden_dim, backend=None, shots=1024, shift=np.pi / 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        # Pass through quantum hybrid layer
        q_out = self.hybrid(x)
        # Convert to probability distribution
        probs = self.softmax(q_out)
        return probs


__all__ = ["FeedForwardClassifier", "QCNet", "QuantumClassifierModel"]
