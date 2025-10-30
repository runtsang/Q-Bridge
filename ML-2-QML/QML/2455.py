"""Hybrid classical‑quantum network for binary classification or regression.

This module implements the same architecture as the classical counterpart
but replaces the dense head with a parameterised quantum circuit.
The quantum sub‑module is built with torchquantum and supports both
parameter‑shift gradient estimation and a general Ry encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import numpy as np

class QuantumHybridHead(tq.QuantumModule):
    """Parameterised quantum circuit that maps a scalar input to a feature vector.

    The circuit consists of a general Ry encoder, a random layer, and
    trainable rotation gates.  Measurement is performed in the Pauli‑Z basis.
    """

    def __init__(self, n_wires: int, shots: int = 1000, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.shots = shots
        self.shift = shift
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, input_vals: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
        """Encode the scalar inputs and return Z‑basis measurement."""
        # input_vals : (batch, 1)
        self.encoder(qdev, input_vals)
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)
        return self.measure(qdev)

class HybridQCNet(nn.Module):
    """Hybrid neural‑quantum architecture mirroring the classical version.

    Parameters
    ----------
    mode : {"classification", "regression"}
        Determines the type of output.  For classification the sigmoid
        activation is applied to the quantum‑derived feature vector.
    n_wires : int
        Number of qubits used in the quantum head.
    shots : int
        Number of shots for the simulator.
    shift : float
        Shift value used in the parameter‑shift rule (default π/2).
    """

    def __init__(
        self,
        mode: str = "classification",
        n_wires: int = 2,
        shots: int = 1000,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.n_wires = n_wires
        # Convolutional feature extractor (identical to the classical version)
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
        self.quantum_head = QuantumHybridHead(n_wires, shots, shift)
        # Linear mapping from quantum features to a scalar output
        self.linear_head = nn.Linear(n_wires, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid network."""
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
        x = self.fc3(x)  # (batch, 1)
        # Prepare quantum device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        q_features = self.quantum_head(x, qdev)  # (batch, n_wires)
        out = self.linear_head(q_features)  # (batch, 1)
        if self.mode == "classification":
            out = torch.sigmoid(out)
            return torch.cat((out, 1 - out), dim=-1)
        return out.squeeze(-1)

__all__ = ["HybridQCNet"]
