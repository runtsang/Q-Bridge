"""Quantum QCNN using Pennylane with a variational ansatz.

The module builds a parameterised QCNN that emulates convolution and pooling
through entangling blocks.  It returns a class ``QCNNModel`` identical to
the classical counterpart; the factory ``QCNN()`` produces an instance
ready for hybrid training with ``torch`` optimisers.
"""

from __future__ import annotations

import pennylane as qml
import torch
import numpy as np
from typing import Tuple


class QCNNModel:
    """Quantum QCNN implemented with Pennylane."""

    def __init__(
        self,
        num_qubits: int = 8,
        conv_depth: int = 3,
        pool_depth: int = 3,
        device: str | qml.Device | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.conv_depth = conv_depth
        self.pool_depth = pool_depth
        self.dev = device or qml.device("default.qubit", wires=num_qubits)

        # Number of variational parameters: 3 per qubit per layer (RZ,RY,CNOT)
        self.num_params = (conv_depth + pool_depth) * self.num_qubits * 3
        self.weights = torch.randn(self.num_params) * 0.1

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Feature map: encode inputs as rotations
            for i in range(self.num_qubits):
                qml.RY(inputs[i], wires=i)

            # Variational layers
            idx = 0
            for _ in range(self.conv_depth):
                # Convolution block: pairwise entanglement
                for q in range(0, self.num_qubits - 1, 2):
                    qml.RZ(weights[idx], wires=q)
                    qml.RY(weights[idx + 1], wires=q + 1)
                    qml.CNOT(wires=[q, q + 1])
                    idx += 2
                # Pooling: rotate and disentangle
                for q in range(0, self.num_qubits - 1, 2):
                    qml.RZ(weights[idx], wires=q)
                    qml.RY(weights[idx + 1], wires=q + 1)
                    qml.CNOT(wires=[q, q + 1])
                    idx += 2

            # Output: expectation of PauliZ on the last qubit
            return qml.expval(qml.PauliZ(self.num_qubits - 1))

        self.circuit = circuit

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the quantum circuit output for a batch of inputs."""
        return torch.stack([self.circuit(inp, self.weights) for inp in inputs])

    def train(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        epochs: int = 200,
    ) -> None:
        """Simple hybrid training loop using Adam optimiser."""
        opt = torch.optim.Adam([self.weights], lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        for _ in range(epochs):
            opt.zero_grad()
            preds = self.predict(data)
            loss = loss_fn(preds, labels)
            loss.backward()
            opt.step()

    def __repr__(self) -> str:
        return f"QCNNModel(num_qubits={self.num_qubits}, conv_depth={self.conv_depth})"


def QCNN() -> QCNNModel:
    """Factory producing the quantum QCNN."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
