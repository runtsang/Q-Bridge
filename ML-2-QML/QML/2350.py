"""Hybrid quantum‑classical model featuring a depth‑parameterized variational circuit
   and a classical post‑processing classifier.

   The model mirrors the structure of the classical counterpart but replaces the
   fully‑connected layers with a PennyLane variational ansatz.  The circuit
   encodes the pooled image features via RX gates, applies `depth` layers of
   Ry rotations followed by CZ entanglement, and measures Pauli‑Z expectation
   values.  These measurement outcomes are fed to a trainable linear layer
   that outputs class logits.
"""

import torch
import torch.nn as nn
import pennylane as qml
import pennylane.numpy as np

class HybridNATModel(nn.Module):
    """
    Hybrid quantum‑classical model combining a PennyLane variational circuit
    with a classical linear classifier.  The architecture is designed to
    replicate the Quantum‑NAT pipeline while allowing depth‑controlled
    expressivity.
    """
    def __init__(self, num_qubits: int = 4, depth: int = 2, num_classes: int = 4):
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.dev = qml.device("default.qubit", wires=num_qubits)

        # Trainable variational parameters
        self.weights = nn.Parameter(torch.randn(depth * num_qubits))

        # Classical post‑processing layer
        self.classifier = nn.Linear(num_qubits, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

        # Build the quantum node
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            # Feature encoding via RX
            for i in range(self.num_qubits):
                qml.RX(inputs[i], wires=i)

            # Variational layers
            idx = 0
            for _ in range(self.depth):
                for i in range(self.num_qubits):
                    qml.RY(weights[idx], wires=i)
                    idx += 1
                for i in range(self.num_qubits - 1):
                    qml.CZ(wires=[i, i + 1])

            # Measurement of Pauli‑Z expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Pool the image to match the number of qubits
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, self.num_qubits)
        # Run the variational circuit
        q_out = self.circuit(pooled, self.weights)
        # Classical classifier
        logits = self.classifier(q_out)
        return self.norm(logits)

__all__ = ["HybridNATModel"]
