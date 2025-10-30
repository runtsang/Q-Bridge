"""Quantum-enhanced quanvolution network using Pennylane variational circuits."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QuanvolutionEnhanced(nn.Module):
    """Hybrid network that replaces the classical filter with a variational quantum circuit."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Quantum device: statevector simulator
        self.device = qml.device("default.qubit", wires=self.n_wires)
        # QNode with batched interface
        self.qnode = qml.QNode(self._quantum_circuit, self.device, interface="torch")
        # Linear head
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def _quantum_circuit(self, x: torch.Tensor) -> torch.Tensor:
        """
        Variational circuit that encodes a 4‑dimensional patch.
        :param x: Tensor of shape (4,) or (batch, 4)
        :return: Expectation values of Pauli‑Z on each wire.
        """
        # Encode each pixel with an Ry rotation
        for i in range(self.n_wires):
            qml.RY(x[i], wires=i)
        # Entangle neighbouring qubits
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[2, 3])
        # Return expectation values
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x: (batch, 1, 28, 28)
        batch_size = x.shape[0]
        x = x.view(batch_size, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, r:r + 2, c:c + 2].view(batch_size, 4)
                # Quantum measurement (batched)
                measurement = self.qnode(patch)  # shape (batch, 4)
                # Residual connection: add raw patch values
                measurement = measurement + patch
                patches.append(measurement)
        features = torch.cat(patches, dim=1)  # shape (batch, 4*14*14)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionEnhanced"]
