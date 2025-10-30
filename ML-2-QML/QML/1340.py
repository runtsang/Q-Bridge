import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuantumLayer(nn.Module):
    """Quantum variational layer using Pennylane with parameter‑shift gradients."""
    def __init__(self, n_qubits: int = 2, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        # Parameters for the variational circuit
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits))

    def circuit(self, params, x):
        # Input encoding: rotate each qubit by the corresponding input value
        for i in range(self.n_qubits):
            qml.RY(x[i], wires=i)
        # Variational layers
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                qml.RY(params[layer, q], wires=q)
            # Entanglement pattern
            for q in range(self.n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        # Expectation value of PauliZ on the first qubit
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_qubits)
        qs = []
        for sample in x:
            # Convert sample to numpy for Pennylane
            exp = qml.QNode(self.circuit, self.dev)(self.params, sample.detach().cpu().numpy())
            qs.append(exp)
        return torch.tensor(qs, dtype=torch.float32, device=x.device)

class HybridBinaryClassifier(nn.Module):
    """Hybrid CNN + quantum variational head for binary classification."""
    def __init__(self, num_classes: int = 2, n_qubits: int = 2, n_layers: int = 2):
        super().__init__()
        # Classical feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_qubits)  # Output to feed into quantum layer

        # Quantum layer
        self.quantum = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers)

        # Final classifier
        self.output = nn.Linear(n_qubits, num_classes)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Quantum variational layer
        q_out = self.quantum(x)
        logits = self.output(q_out)
        probs = self.activation(logits)
        return probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.forward(x)
        return torch.argmax(probs, dim=-1)

__all__ = ["HybridBinaryClassifier", "QuantumLayer"]
