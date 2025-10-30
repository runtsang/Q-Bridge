"""Quantum‑enhanced variant of QuantumNAT using Pennylane.

The model encodes a 4‑dimensional classical embedding into a 4‑qubit
variational circuit.  The circuit is defined with a trainable ansatz
and executed on Pennylane's default.qubit simulator.  The output
consists of the expectation values of Pauli‑Z on each qubit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from typing import Tuple


class QuantumNATEnhanced(nn.Module):
    """Quantum module that fuses a classical pooling step with a variational circuit.

    The model accepts a batch of 28×28 grayscale images, reduces them to a
    four‑dimensional feature vector via average pooling, and feeds each
    vector into a 4‑qubit variational circuit.  The circuit is defined
    with a repeatable ansatz and entangling CNOTs.  The device is
    Pennylane's lightweight `default.qubit` simulator.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Parameters for the encoding (Ry,Rz) and the variational ansatz
        self.encoder_params = nn.Parameter(torch.randn(n_qubits, 2))
        self.variational_params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))

        self.batchnorm = nn.BatchNorm1d(n_qubits)

        # Compile the qnode once
        self.qnode = qml.qnode(self.dev, interface="torch", diff_method="backprop")(
            self._qcircuit
        )

    def _qcircuit(self, x: torch.Tensor) -> Tuple[torch.Tensor,...]:
        """Variational circuit that returns Pauli‑Z expectations for each qubit."""
        # Encoding
        for q in range(self.n_qubits):
            qml.RY(self.encoder_params[q, 0] * x[q], wires=q)
            qml.RZ(self.encoder_params[q, 1] * x[q], wires=q)

        # Ansatz layers
        for i in range(self.n_layers):
            for q in range(self.n_qubits):
                qml.RY(self.variational_params[i, q, 0], wires=q)
                qml.RZ(self.variational_params[i, q, 1], wires=q)
                qml.RY(self.variational_params[i, q, 2], wires=q)
            # Entangling CNOT chain
            for q in range(self.n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])

        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid quantum model."""
        # Classical pooling: average pool to 4×4 blocks and reduce to 4 features
        pooled = F.avg_pool2d(x, 6).view(x.shape[0], -1)  # shape (batch, 16)
        features = pooled.view(x.shape[0], 4, 4).mean(dim=2)  # shape (batch, 4)

        # Execute the qnode on each feature vector
        out = torch.stack([self.qnode(feat) for feat in features], dim=0)
        return self.batchnorm(out)
