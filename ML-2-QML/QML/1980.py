import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np

class QuantumHybridClassifier(nn.Module):
    """
    CNN backbone followed by a depth‑controlled variational quantum circuit head.
    """
    def __init__(self, in_features: int = 55815, depth: int = 3, shift: float = np.pi / 2) -> None:
        super().__init__()
        # CNN backbone identical to the seed
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Quantum circuit parameters
        self.depth = depth
        self.shift = shift
        self.dev = qml.device('default.qubit', wires=1)
        self.params = nn.Parameter(torch.randn(depth * 2))
        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def quantum_node(angles, params):
            # Encode input angle(s) as RY rotations
            for a in angles:
                qml.RY(a, wires=0)
            # Variational ansatz
            for d in range(depth):
                qml.RZ(params[2 * d], wires=0)
                qml.RY(params[2 * d + 1], wires=0)
            return qml.expval(qml.PauliZ(0))
        self.quantum_node = quantum_node

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)
        # Compute quantum expectation for each batch element
        expectations = []
        for angle in x:
            exp = self.quantum_node(angle, self.params)
            expectations.append(exp)
        expectations = torch.stack(expectations)
        logits = expectations
        probs = torch.sigmoid(logits + self.shift)
        return torch.stack([probs, 1 - probs], dim=-1)
