"""Hybrid quantum‑classical model using PennyLane.

Key components:
* Angle‑encoding of classical features into 4 qubits.
* 3‑layer parameterised circuit with RX/RZ rotations and CNOT entanglement.
* Expectation values of PauliZ are used as features for a linear classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pennylane as qml
import pennylane.numpy as np
from torch import Tensor
from typing import Any


class HybridQuantumNATModel(nn.Module):
    """Quantum‑classical hybrid model with a variational circuit."""

    def __init__(self, n_layers: int = 3, n_wires: int = 4) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_wires = n_wires

        # Classical pre‑processing to bring the 16‑dim input down to 4
        self.pre_fc = nn.Linear(16, self.n_wires)

        # Quantum device: 4 qubits, 1024 shots for expectation estimation
        self.dev = qml.device("default.qubit", wires=self.n_wires, shots=1024)

        # Parameterised weights for the variational circuit
        self.weights = nn.Parameter(
            torch.randn(self.n_layers, self.n_wires, 2, requires_grad=True)
        )

        # Linear layer mapping quantum observables to logits
        self.fc = nn.Linear(self.n_wires, 4)
        self.norm = nn.BatchNorm1d(4)

        # Define the quantum node with Torch interface
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: Tensor, weights: Tensor) -> Tensor:
            # Angle‑encoding of classical inputs
            for w in range(self.n_wires):
                qml.RX(inputs[w], wires=w)
            # Variational layers
            for layer in range(self.n_layers):
                for w in range(self.n_wires):
                    qml.RX(weights[layer, w, 0], wires=w)
                    qml.RZ(weights[layer, w, 1], wires=w)
                # Entanglement pattern: a ring of CNOTs
                for w in range(self.n_wires):
                    qml.CNOT(wires=[w, (w + 1) % self.n_wires])
            # Expectation values of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_wires)]

        self.circuit = circuit

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # Classical preprocessing: 28x28 images --> 16‑dim feature vector
        bsz = x.shape[0]
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, -1)  # (bsz, 16)
        encoded = self.pre_fc(pooled)  # (bsz, n_wires)
        # Run the variational circuit in batch mode
        q_out = self.circuit(encoded, self.weights)  # (bsz, n_wires)
        logits = self.fc(q_out)
        return self.norm(logits)


__all__ = ["HybridQuantumNATModel"]
