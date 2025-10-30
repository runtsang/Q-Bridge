"""Quantum Quanvolution implementation using Pennylane."""
from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any


class _QuantumFilter(nn.Module):
    """
    Variational quantum filter operating on 2×2 patches.

    Each patch is encoded into two qubits via RY rotations.
    A small parameter‑shared circuit of depth `num_layers` is applied,
    and the expectation values of Pauli‑Z are returned as features.
    """
    def __init__(self, wires: int = 2, num_layers: int = 2, shots: int = 1024) -> None:
        super().__init__()
        self.wires = wires
        self.num_layers = num_layers
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=self.wires, shots=self.shots)

        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Encode the 2×2 patch (4 values) into two qubits
            # Here we use the first two values; the remaining are ignored
            qml.RY(x[0], wires=0)
            qml.RY(x[1], wires=1)
            # Variational circuit
            for _ in range(self.num_layers):
                for i in range(self.wires):
                    qml.RZ(params[i], wires=i)
                qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        # Initialise parameters
        self.params = nn.Parameter(torch.rand(self.wires))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over a batch of images.
        Returns a tensor of shape (N, 14*14) where each entry is a quantum measurement.
        """
        N, H, W = x.shape
        outputs = []

        for r in range(0, H, 2):
            for c in range(0, W, 2):
                patch = x[:, r:r+2, c:c+2].reshape(N, -1)  # (N, 4)
                # Use only first two values for encoding
                patch = patch[:, :2]
                # Normalize to [0, pi]
                patch = torch.pi * patch / (patch.max() + 1e-8)
                # Compute measurement for each sample
                meas = self.circuit(patch, self.params).reshape(-1)
                outputs.append(meas)

        # Stack all patch measurements
        out = torch.stack(outputs, dim=1)  # (N, 14*14)
        return out


class Quanvolution(nn.Module):
    """
    Quantum quanvolution model.

    The filter uses a parameter‑shared variational circuit per patch.
    The resulting 14×14 feature map is fed into a linear classifier.
    """
    def __init__(self, num_classes: int = 10, num_layers: int = 2, shots: int = 1024) -> None:
        super().__init__()
        self.filter = _QuantumFilter(num_layers=num_layers, shots=shots)
        feature_dim = 14 * 14  # 2×2 patches on 28×28 image
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["Quanvolution"]
