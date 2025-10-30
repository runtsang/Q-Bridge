"""Quantum hybrid binary classifier using PennyLane.

This module implements a variational quantum circuit as the head of a
convolutional backbone.  The circuit uses a parameter‑shift rule for
gradient computation, enabling end‑to‑end training with PyTorch.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumLayer(nn.Module):
    """Parameter‑shifted variational layer returning a single expectation."""

    def __init__(self, n_qubits: int = 2, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.shift = shift
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(params: torch.Tensor) -> torch.Tensor:
            for i in range(n_qubits):
                qml.RY(params[i], wires=i)
            # Simple entanglement pattern
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.Z(0))

        self.circuit = circuit

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Return the circuit expectation for a batch of 1‑D parameters."""
        # Ensure params shape (batch, n_qubits)
        return self.circuit(params)


class HybridBinaryClassifier(nn.Module):
    """Hybrid classifier that replaces the MLP head with a quantum layer."""

    def __init__(self, shift: float = 0.0, n_qubits: int = 2) -> None:
        super().__init__()
        self.shift = shift
        self.n_qubits = n_qubits

        # Residual backbone identical to the classical counterpart
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.flatten = nn.Flatten()
        # Quantum head
        self.quantum_head = QuantumLayer(n_qubits=n_qubits, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities for a batch of images."""
        x = self.features(x)
        x = self.flatten(x)
        # Map 64‑dim feature vector to n_qubits parameters
        params = nn.Linear(64, self.n_qubits)(x)
        logits = self.quantum_head(params)
        probs = torch.sigmoid(logits + self.shift)
        return probs


__all__ = ["HybridBinaryClassifier"]
