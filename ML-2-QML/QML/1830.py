"""Quantum model extending Quantum‑NAT using PennyLane.

The circuit encodes a 4‑element feature vector via an
AngleEmbedding, applies a variational ansatz with RY, CNOT,
and RZ gates, measures Pauli‑Z expectation values, and
feeds them into a classical MLP head. The design is
fully differentiable and can be run on simulators or
real devices via PennyLane.
"""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

# Quantum device
dev = qml.device("default.qubit", wires=4)

def variational_ansatz(params, wires):
    """Parametrized rotations followed by entanglement."""
    for i, wire in enumerate(wires):
        qml.RY(params[i], wires=wire)
    for i in range(len(wires) - 1):
        qml.CNOT([wires[i], wires[i + 1]])
    for i, wire in enumerate(wires):
        qml.RZ(params[len(wires) + i], wires=wire)

@qml.qnode(dev, interface="torch")
def quantum_circuit(x, params):
    """Full circuit: feature map + ansatz + measurement."""
    qml.AngleEmbedding(x, wires=range(4))
    variational_ansatz(params, wires=range(4))
    return [qml.expval(qml.PauliZ(w)) for w in range(4)]

class QuantumNATPlus(nn.Module):
    """Hybrid quantum‑classical model with a variational layer."""

    def __init__(self, num_classes: int = 4, n_params: int = 8) -> None:
        super().__init__()
        # Quantum parameters
        self.q_params = nn.Parameter(torch.randn(n_params))
        # Classical classifier
        self.classifier = nn.Sequential(
            nn.Linear(n_params, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: average‑pool to a 4‑element vector, run quantum
        circuit for each sample, then classify.
        """
        batch_size = x.shape[0]
        # Average‑pool to a 4‑element vector
        pooled = F.avg_pool2d(x, kernel_size=6, stride=6).view(batch_size, -1)
        features = pooled[:, :4]
        # Run quantum circuit for each sample
        q_out = []
        for i in range(batch_size):
            q_out.append(quantum_circuit(features[i], self.q_params))
        q_out = torch.stack(q_out)
        logits = self.classifier(q_out)
        return self.norm(logits)

__all__ = ["QuantumNATPlus"]
