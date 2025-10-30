"""Quantum variant of the NAT model using Pennylane and a variational ansatz."""

import pennylane as qml
import torch
import torch.nn as nn
import numpy as np


class QuantumNATEnhanced:
    """Hybrid quantum circuit that encodes a 4‑dim feature vector into 4 qubits."""

    def __init__(self, n_wires: int = 4, device: str = "default.qubit", shots: int = 1024) -> None:
        self.n_wires = n_wires
        self.device = qml.device(device, wires=n_wires, shots=shots, backend="torch")
        # Trainable parameters for a two‑layer ansatz
        self.params = torch.nn.Parameter(torch.rand(2, n_wires, 3))
        self.circuit = qml.QNode(self._circuit, self.device, interface="torch")

    def _circuit(self, data: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Data‑dependent encoding followed by a depth‑2 variational ansatz."""
        # Encode each data element into a rotation about Y
        for i, w in enumerate(range(self.n_wires)):
            qml.RY(data[i], wires=w)
        # Variational layers
        for layer in range(params.shape[0]):
            for w in range(self.n_wires):
                qml.RY(params[layer, w, 0], wires=w)
                qml.RZ(params[layer, w, 1], wires=w)
                qml.RX(params[layer, w, 2], wires=w)
            # Entangle neighbouring qubits
            for w in range(self.n_wires - 1):
                qml.CNOT(wires=[w, w + 1])
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), \
               qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 4) – feature vector to encode.
        Returns:
            Tensor of shape (batch, 4) – normalized expectation values.
        """
        batch, _ = x.shape
        out = torch.stack([self.circuit(x[i], self.params) for i in range(batch)], dim=0)
        norm = nn.BatchNorm1d(4)
        return norm(out)

__all__ = ["QuantumNATEnhanced"]
