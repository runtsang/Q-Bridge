"""HybridNATModel: Quantum‑enhanced architecture using Qiskit.

This QML variant mirrors the ML version but replaces the Pennylane quantum
feature extractor with a Qiskit circuit wrapped by TorchConnector.  The
class contains a residual CNN backbone, a fully‑connected head, and a
Qiskit‑based variational circuit that outputs four expectation values
which are fused with the classical output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
import qiskit
from qiskit.circuit.library import StronglyEntanglingLayers
from qiskit.circuit.library import StronglyEntanglingLayersEntanglement
from qiskit_machine_learning.connectors import TorchConnector


class ResidualBlock(nn.Module):
    """Resemble the residual block from the ML implementation."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels!= out_channels else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class QuantumFeatureExtractor(nn.Module):
    """Qiskit‑based variational circuit wrapped for PyTorch."""
    def __init__(self, n_wires: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        # Build a strongly entangling circuit with n_wires qubits
        circuit = StronglyEntanglingLayers(
            num_qubits=n_wires,
            reps=n_layers,
            entanglement=StronglyEntanglingLayersEntanglement.full,
            insert_barriers=False,
        )
        # The first 4 parameters of the circuit will be set
        # by the input vector (angle embedding)
        self.torch_connector = TorchConnector(
            circuit, input_params=[circuit.params[0:4]]
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x shape: (batch, 4)
        return self.torch_connector(x)


class HybridNATModel(nn.Module):
    """Quantum‑enhanced hybrid model using Qiskit for the quantum part."""
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
        qfeat = self.qextractor(x)          # quantum‑encoded features
        out = x + qfeat                    # residual fusion
        return self.norm(out)


__all__ = ["HybridNATModel"]
