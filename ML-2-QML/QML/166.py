"""Quantum circuit for fraud detection using Pennylane.

The circuit maps a classical feature vector to a probability
estimate via a simple variational ansatz.
"""

import pennylane as qml
import torch

# Use a 2‑qubit device; the number of wires can be adapted
dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev, interface="torch")
def quantum_layer(features: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    Variational circuit that takes a feature vector and a set of
    circuit parameters and returns a probability estimate.
    """
    # Map each feature to an RX rotation on its corresponding qubit
    for i, feat in enumerate(features):
        qml.RX(feat, wires=i)
    # Entangling layer with parameterized rotations
    for i in range(len(params)):
        qml.CNOT(wires=[i, (i + 1) % 2])
        qml.RZ(params[i], wires=i)
    # Measurement: expectation value of PauliZ on qubit 0
    return qml.expval(qml.PauliZ(0))
