"""Quantum‑enhanced binary classifier using Pennylane."""

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Device with 4 qubits
dev = qml.device("default.qubit", wires=4)

class QuantumLayer(nn.Module):
    """Parameterized quantum circuit producing expectation of Z."""
    def __init__(self, num_wires: int = 4, shift: float = np.pi / 2):
        super().__init__()
        self.num_wires = num_wires
        self.shift = shift
        # Learnable parameters for the circuit (unused in this simple example but kept for extensibility)
        self.params = nn.Parameter(torch.randn(num_wires))

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(self, inputs):
        for w in range(self.num_wires):
            qml.RY(inputs[w], wires=w)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[2, 3])
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_wires)
        expectations = torch.stack([self.circuit(sample) for sample in x])
        return expectations.unsqueeze(-1)  # (batch, 1)

class HybridClassifier(nn.Module):
    """Quantum‑enhanced binary classifier."""
    def __init__(self, num_features: int = 4, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.quantum = QuantumLayer(num_wires=num_features)
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_out = self.quantum(x)  # (batch, 1)
        logits = self.classifier(q_out)
        probs = F.softmax(logits, dim=-1)
        return probs

__all__ = ["QuantumLayer", "HybridClassifier"]
