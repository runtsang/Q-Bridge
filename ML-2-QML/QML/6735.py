import torch
import pennylane as qml
from pennylane import numpy as pnp
from torch import nn
from typing import Tuple

def _build_ansatz(num_qubits: int, reps: int = 3) -> qml.QNode:
    """Return a parameterized circuit used as the quantum latent layer."""
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor):
        # Encode inputs into rotations
        for i in range(num_qubits):
            qml.RX(inputs[i], wires=i)
            qml.RY(inputs[i], wires=i)
        # Trainable entangling layer
        for r in range(reps):
            for i in range(num_qubits):
                qml.CNOT(wires=[i, (i + 1) % num_qubits])
                qml.RZ(weights[r * num_qubits + i], wires=i)
        # Return expectation of Z on first qubit as output
        return qml.expval(qml.PauliZ(0))
    return circuit

class QNNWrapper(nn.Module):
    """Torch‑compatible wrapper around a PennyLane QNode."""
    def __init__(self, num_qubits: int, reps: int = 3):
        super().__init__()
        self.num_qubits = num_qubits
        self.reps = reps
        # Trainable weights
        self.weights = nn.Parameter(pnp.random.randn(reps * num_qubits))
        self.circuit = _build_ansatz(num_qubits, reps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.circuit(inputs, self.weights)

def Autoencoder_QML(num_qubits: int = 4, reps: int = 3) -> QNNWrapper:
    """Return a quantum neural network that can replace the latent layer."""
    return QNNWrapper(num_qubits, reps)

__all__ = ["QNNWrapper", "Autoencoder_QML"]
