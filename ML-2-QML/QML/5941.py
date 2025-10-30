"""QuantumNATGen – quantum branch using PennyLane."""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn


class QFCModel(nn.Module):
    """
    A variational quantum circuit that maps a flattened feature vector to a 4‑dimensional embedding.
    Uses 4 qubits and a few layers of parametrized rotations and CNOTs.
    """

    def __init__(
        self,
        input_dim: int = 64,
        n_qubits: int = 4,
        n_layers: int = 2,
        device: str | torch.device = 'cpu',
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.qnode(
            self.dev, interface="torch", diff_method="backprop"
        )(self._circuit)
        # Trainable ansatz parameters
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))

    def _circuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # Encode classical data into qubit rotations
        for i in range(self.input_dim):
            qml.RY(x[i], wires=i % self.n_qubits)
        # Variational ansatz
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                qml.Rot(
                    params[layer, q, 0],
                    params[layer, q, 1],
                    params[layer, q, 2],
                    wires=q,
                )
            # Entangling layer
            for q in range(self.n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])
        # Measurement
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x should be a 1‑D tensor of length input_dim.
        Returns a 4‑dimensional embedding.
        """
        return self.qnode(x, self.params)
