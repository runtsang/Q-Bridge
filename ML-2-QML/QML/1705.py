"""Hybrid quantum‑CPU binary classifier that uses a two‑qubit variational circuit
as the final layer.

The model mirrors the classical version but replaces the quantum‑inspired
head with a true quantum circuit implemented with Pennylane.  A linear
projection maps the pre‑logit to two rotation angles, and the expectation
value of Pauli‑Z on the first qubit is used as the logit.  Gradients are
computed via Pennylane’s parameter‑shift rule, making the layer fully
differentiable in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane.qnn import TorchLayer

# Quantum device with two qubits
dev = qml.device("default.qubit", wires=2)


def variational_ansatz(params, wires):
    qml.Hadamard(wires=wires[0])
    qml.Hadamard(wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])


@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_circuit(params):
    variational_ansatz(params, wires=[0, 1])
    return qml.expval(qml.PauliZ(0))


class QuantumHybridCNNClassifier(nn.Module):
    """CNN followed by a two‑qubit variational quantum head."""
    def __init__(self) -> None:
        super().__init__()
        # Convolutional front‑end
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Fully connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Map the single logit to two rotation angles
        self.angle_mapper = nn.Linear(1, 2)
        # Quantum layer
        self.q_layer = TorchLayer(quantum_circuit, output_dim=1, interface="torch", device=dev)

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
        x = self.fc3(x)                # (batch, 1)
        angles = self.angle_mapper(x)  # (batch, 2)
        q_out = self.q_layer(angles)   # (batch, 1)
        prob = torch.sigmoid(q_out)
        return torch.cat((prob, 1 - prob), dim=-1)


__all__ = ["QuantumHybridCNNClassifier"]
