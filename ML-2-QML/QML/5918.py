"""QuantumHybridClassifier – quantum‑enhanced binary classifier.

This module builds upon the classical backbone and replaces the
final linear layer with a variational quantum circuit.  The
circuit is parameterised by the output of a fully‑connected layer
and produces an expectation value that is converted to a
probability via a sigmoid.  The design demonstrates how a
classical neural network can be coupled to a quantum
variational layer while retaining end‑to‑end differentiability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np

__all__ = ["QuantumHybridClassifier"]


# Define a simple two‑qubit variational circuit
def quantum_circuit(params: np.ndarray) -> np.ndarray:
    """Return the expectation of Z on the first qubit."""
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(params[2], wires=0)
    qml.RZ(params[3], wires=1)
    return qml.expval(qml.PauliZ(0))


class QuantumHybridClassifier(nn.Module):
    """Convolutional network with a variational quantum head."""
    def __init__(self, num_params: int = 4, device_name: str = "default.qubit") -> None:
        super().__init__()
        # Classical backbone identical to FeatureExtractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Linear layer to produce parameters for the quantum circuit
        self.quantum_params = nn.Linear(1, num_params)
        # Quantum device
        self.dev = qml.device(device_name, wires=2)
        # QNode wrapped as a Torch layer
        qnode = qml.QNode(quantum_circuit, self.dev, interface="torch")
        self.qlayer = qml.qnn.TorchLayer(qnode, weights_shape=(num_params,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Produce parameters for the quantum circuit
        q_params = self.quantum_params(x)
        # Compute quantum expectation
        expectation = self.qlayer(q_params)
        prob = torch.sigmoid(expectation)
        return torch.cat([prob, 1 - prob], dim=-1)
