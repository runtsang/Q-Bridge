"""Hybrid quantum‑classical binary classifier using Pennylane.

The quantum head is a parameterised variational circuit with a
parameter‑shift gradient.  The rest of the network is identical to
the classical version, allowing direct comparison of training
dynamics.
"""

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

# Two‑qubit device with a moderate shot count for gradients
dev = qml.device("default.qubit", wires=2, shots=1000)

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_expectation(param: torch.Tensor) -> torch.Tensor:
    """Single‑parameter variational circuit whose expectation value
    is used as the quantum classification head."""
    qml.RY(param, wires=0)
    qml.RY(param, wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

class HybridQCNet(nn.Module):
    """Quantum‑backed binary classifier.  The final dense layer feeds
    the quantum circuit; the circuit's expectation value is turned into
    a probability via a sigmoid."""
    def __init__(self,
                 in_features: int = 55815,
                 hidden_sizes: tuple[int, int] = (120, 84),
                 dropout: tuple[float, float] = (0.2, 0.5)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=dropout[0])
        self.drop2 = nn.Dropout2d(p=dropout[1])
        self.fc1 = nn.Linear(in_features, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)  # one output for the quantum angle
        self._sigmoid = nn.Sigmoid()

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
        x = self.fc3(x)          # shape (..., 1)
        params = x.squeeze(-1)   # shape (...,)
        probs = []
        for p in params:
            exp_val = quantum_expectation(p)
            probs.append(self._sigmoid(exp_val))
        probs = torch.stack(probs)          # shape (batch,)
        probs = probs.unsqueeze(-1)         # shape (batch, 1)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridQCNet"]
