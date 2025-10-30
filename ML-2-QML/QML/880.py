import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

class QuantumLayer(nn.Module):
    """Quantum layer that maps a scalar to a probability via a 4‑qubit variational circuit."""
    def __init__(self, num_qubits: int = 4, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.shift = shift
        self.dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(theta: torch.Tensor):
            # Apply the same rotation to all qubits
            for i in range(num_qubits):
                qml.RY(theta, wires=i)
            # Entangle qubits with a simple CNOT ladder
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measure expectation of Pauli‑Z on the first qubit
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        # theta shape (batch, 1)
        probs = []
        for i in range(theta.size(0)):
            probs.append(self.circuit(theta[i]))
        return torch.stack(probs).unsqueeze(-1)  # (batch, 1)

class HybridClassifier(nn.Module):
    """CNN followed by a variational quantum layer for binary classification."""
    def __init__(self, in_channels: int = 3, num_qubits: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(32, 1)
        self.quantum = QuantumLayer(num_qubits=num_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probs = self.quantum(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridClassifier", "QuantumLayer"]
