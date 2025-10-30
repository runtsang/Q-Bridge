"""QuantumHybridClassifier – quantum‑classical binary classifier using Pennylane.

This module implements a hybrid network that replaces the dense head of the
original architecture with a variational quantum circuit.  The circuit is
parameterised by the output of a small classical feed‑forward head and
evaluated on a Pennylane device.  The expectation value of Pauli‑Z is
treated as a learnable quantum activation, optionally shifted by a
learnable bias.  The design supports end‑to‑end training with
automatic differentiation and can be swapped with the classical surrogate
for ablation studies.
"""

import pennylane as qml
import pennylane.numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Quantum device – default qubit simulator with 100 shots
dev = qml.device("default.qubit", wires=1, shots=100)


def _quantum_layer(theta: torch.Tensor) -> torch.Tensor:
    """Variational circuit that returns the expectation of Pauli‑Z."""
    qml.RY(theta, wires=0)
    return qml.expval(qml.PauliZ(0))


# Wrap the circuit in a QNode that accepts Torch tensors
qnode = qml.QNode(_quantum_layer, dev, interface="torch")


class QuantumHybridClassifier(nn.Module):
    """Hybrid CNN + variational quantum head for binary classification.

    Parameters
    ----------
    in_features : int
        Size of the flattened feature vector from the CNN backbone.
    hidden_dim : int, default 32
        Width of the intermediate linear layer before the quantum head.
    shift : float, default 0.0
        Initial bias added to the quantum expectation before the sigmoid.
    """

    def __init__(self, in_features: int, hidden_dim: int = 32, shift: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.shift = nn.Parameter(torch.tensor(shift, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing a two‑class probability distribution."""
        x = F.relu(self.fc1(x))
        logits = self.fc2(x).squeeze(-1)  # shape: (batch,)
        # Quantum expectation
        q_expect = qnode(logits)
        probs = torch.sigmoid(q_expect + self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuantumHybridClassifier"]
