"""Quantum hybrid classifier.

This module implements a convolutional backbone followed by a quantum
circuit head.  The circuit is parameterized by a random layer and
trainable single‑qubit rotations.  A measurement of Pauli‑Z on each
wire produces a feature vector that is fed into a classical linear
classifier.  The overall architecture mirrors the classical
HybridClassifier but replaces the hybrid head with a true quantum
expectation value.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import numpy as np


class QuantumHybridHead(tq.QuantumModule):
    """Quantum head that encodes a scalar into a set of qubits, applies a
    parameter‑dependent circuit and measures Pauli‑Z.
    """
    def __init__(self, n_wires: int, shift: float = np.pi / 2, shots: int = 1024):
        super().__init__()
        self.n_wires = n_wires
        self.shift = shift
        self.shots = shots
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{n_wires}xRy"]
        )
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires, 1)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.random_layer(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


class HybridClassifier(tq.QuantumModule):
    """Convolutional backbone followed by a QuantumHybridHead."""
    def __init__(self, shift: float = np.pi / 2, shots: int = 1024) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        dummy = torch.zeros(1, 3, 32, 32)
        x = self._forward_conv(dummy)
        flat_size = x.view(1, -1).size(1)
        self.fc1 = nn.Linear(flat_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.head = QuantumHybridHead(n_wires=self.fc3.out_features,
                                      shift=shift, shots=shots)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.head.n_wires, bsz=bsz, device=x.device)
        encoded = x.repeat(1, self.head.n_wires)
        self.head.encoder(qdev, encoded)
        out = self.head(qdev)
        return torch.cat((out, 1 - out), dim=-1)


__all__ = ["HybridClassifier", "QuantumHybridHead"]
