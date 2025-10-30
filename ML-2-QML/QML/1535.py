"""Quantum hybrid network that applies a parameterised variational circuit to each
2×2 patch of the input image using PennyLane.  The circuit uses angle‑encoding,
entangling layers and measures expectation values of Pauli‑Z on each qubit.
The concatenated measurements form a feature vector for a classical linear head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QuanvolutionHybrid(nn.Module):
    """
    Quantum hybrid network that applies a parameterised variational circuit to each
    2×2 patch of the input image. The circuit uses angle‑encoding, two‑qubit
    entangling layers, and measures the expectation of Pauli‑Z on each qubit.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, n_layers: int = 3, entanglement: str = "full") -> None:
        super().__init__()
        self.n_qubits = 4  # one qubit per pixel in a 2×2 patch
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        self.n_layers = n_layers
        self.entanglement = entanglement

        # Variational ansatz
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Angle encoding
            for i, val in enumerate(inputs):
                qml.RY(val, wires=i)

            # Variational layers
            for layer in range(self.n_layers):
                for qubit in range(self.n_qubits):
                    qml.RY(weights[layer, qubit], wires=qubit)
                if self.entanglement == "full":
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                else:  # star entanglement
                    qml.CNOT(wires=[0, 1])
                    qml.CNOT(wires=[0, 2])
                    qml.CNOT(wires=[0, 3])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit
        self.weights = nn.Parameter(torch.randn(self.n_layers, self.n_qubits))

        # Classical head
        self.linear = nn.Linear(self.n_qubits * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(batch_size, 28, 28)
        patch_features = []

        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, r : r + 2, c : c + 2]  # shape (batch, 2, 2)
                patch_flat = patch.view(batch_size, -1)  # shape (batch, 4)
                meas = self.circuit(patch_flat, self.weights)
                patch_features.append(meas)

        features = torch.cat(patch_features, dim=1)  # shape (batch, 4*14*14)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
