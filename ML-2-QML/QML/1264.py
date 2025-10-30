import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import pennylane.numpy as np

class QuantumCircuit(nn.Module):
    """Variational circuit that outputs the expectation of Pauli‑Z on qubit 0."""
    def __init__(self, num_qubits: int, depth: int = 2):
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(params, x):
            # Encode classical data with Ry rotations
            for i, xi in enumerate(x):
                qml.RY(xi, wires=i)
            # Variational layers
            for _ in range(depth):
                qml.templates.StronglyEntanglingLayers(params, wires=range(num_qubits))
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        self.params = nn.Parameter(torch.randn(depth, num_qubits, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.circuit(self.params, x)

class QuantumHybridClassifier(nn.Module):
    """CNN backbone followed by a variational quantum circuit head."""
    def __init__(self, num_classes: int = 2, feature_dim: int = 8,
                 num_qubits: int = 2, depth: int = 2):
        super().__init__()
        # feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, feature_dim)
        self.projection = nn.Identity() if feature_dim == num_qubits else nn.Linear(feature_dim, num_qubits)
        self.quantum_head = QuantumCircuit(num_qubits, depth)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        x = self.projection(x)
        z = self.quantum_head(x)
        prob = torch.sigmoid(z)
        return torch.cat([prob, 1 - prob], dim=-1)

__all__ = ["QuantumHybridClassifier"]
