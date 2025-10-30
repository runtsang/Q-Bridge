"""HybridNATModel: Extended classical model with residual CNN and quantum-inspired feature fusion.

This module builds upon the original QFCModel by adding a residual convolutional backbone,
dropout regularisation, and a quantum-inspired feature extractor implemented with
Pennylane's qnode. The design keeps the output dimensionality unchanged (4 features)
while enriching the representation via parameterised quantum circuits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from torch import Tensor


class ResidualBlock(nn.Module):
    """A simple 2‑D residual block with optional channel adjustment."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels!= out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class QuantumFeatureExtractor(nn.Module):
    """Parameterised quantum circuit implemented with Pennylane.

    The circuit embeds the 4‑dimensional classical feature vector into a
    4‑qubit system, applies a stack of strongly entangling layers,
    and returns the expectation values of Pauli‑Z on each wire.
    """

    def __init__(self, n_wires: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_wires)

        # Initialise parameters for the variational layers
        self.params = nn.Parameter(
            torch.randn(n_layers, n_wires, 3, dtype=torch.float64)
        )

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: Tensor, params: Tensor) -> Tensor:
            # Angle embedding of the classical input
            qml.AngleEmbedding(inputs, wires=range(n_wires))
            # Variational layers
            for layer_idx in range(n_layers):
                qml.templates.StronglyEntanglingLayers(
                    params[layer_idx], wires=range(n_wires)
                )
            # Measure expectation values of Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

        self.circuit = circuit

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x is expected to have shape (batch, 4)
        return self.circuit(x, self.params)


class HybridNATModel(nn.Module):
    """Hybrid classical‑quantum model inspired by Quantum‑NAT.

    The architecture comprises:
        * Residual CNN backbone (2 conv layers + residual block)
        * Dropout and batch‑norm regularisation
        * Fully‑connected head producing 4‑dimensional output
        * QuantumFeatureExtractor that transforms the same 4‑dimensional vector
          and its output is added residually
        * Final BatchNorm1d for feature‑wise normalisation
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ResidualBlock(8, 16),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 4),
        )
        self.qextractor = QuantumFeatureExtractor(n_wires=4, n_layers=2)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.fc(x)
        qfeat = self.qextractor(x)  # quantum‑encoded features
        out = x + qfeat  # residual fusion
        return self.norm(out)


__all__ = ["HybridNATModel"]
