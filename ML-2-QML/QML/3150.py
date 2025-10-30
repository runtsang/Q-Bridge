"""Hybrid quantum‑classical binary classifier with a random‑layer quantum head.

The quantum head is built on torchquantum and combines a random layer, trainable
RX/RY gates and a Pauli‑Z measurement, providing expressive feature extraction
while remaining efficient on simulators.  The convolutional backbone is
identical to the classical counterpart.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuantumHybridHead(tq.QuantumModule):
    """Quantum head that applies a random layer and trainable RX/RY gates."""
    def __init__(self, n_wires: int = 1, shift: float = 0.0) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, 1) containing the scalar output of the
            preceding classical head.
        """
        bsz = inputs.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=inputs.device)
        # Randomised feature generation
        self.random_layer(qdev)
        # Encode the classical scalar as a rotation on the single qubit
        # The rotation angle is taken directly from the input tensor.
        self.rx(qdev, wires=0, params=inputs[:, 0])
        self.ry(qdev, wires=0, params=inputs[:, 0])
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

class QCNet(nn.Module):
    """Convolutional network followed by a quantum expectation head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = QuantumHybridHead(n_wires=1, shift=0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
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
        # Quantum hybrid head expects a tensor of shape (batch, 1)
        x = torch.sigmoid(self.hybrid(x))
        return torch.cat((x, 1 - x), dim=-1)

__all__ = ["QuantumHybridHead", "QCNet"]
