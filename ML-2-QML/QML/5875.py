"""
HybridClassifier: Quantum head built with PennyLane that takes the output of a CNN backbone
and returns a probability distribution over two classes.
"""

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridClassifier(nn.Module):
    """
    Quantum CNN backbone followed by a differentiable variational circuit.
    """
    def __init__(self, n_qubits: int = 2, n_classes: int = 2, hidden_dim: int = 32):
        super().__init__()
        # Convolutional backbone (identical to the classical version)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
            nn.Flatten()
        )
        # Compute feature size after conv layers
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            dummy = self.backbone(dummy)
            dummy = dummy.view(dummy.size(0), -1)
            feature_size = dummy.size(1)
        self.fc1 = nn.Linear(feature_size, 120)
        self.fc2 = nn.Linear(120, 84)

        # Quantum device and parameters
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.q_params = nn.Parameter(torch.randn(n_qubits))
        # Classical post‑processing layer
        self.post = nn.Linear(n_qubits, n_classes)

        # Define a PennyLane QNode
        self.qnode = qml.QNode(self._quantum_circuit, device=self.dev, interface="torch")

    def _quantum_circuit(self, params, x):
        """
        Variational circuit that maps a scalar input x to a vector of
        Pauli‑Z expectation values.  The circuit is fully differentiable
        thanks to PennyLane’s autograd integration.
        """
        for i in range(len(params)):
            qml.RY(params[i] * x, wires=i)
        qml.CNOT(wires=[0, 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(len(params))]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        # Use the mean of the features as a scalar input for the circuit
        x_scalar = torch.mean(x, dim=1)
        q_out = self.qnode(self.q_params, x_scalar)  # shape: (batch, n_qubits)
        logits = self.post(q_out)
        return F.softmax(logits, dim=-1)  # shape: (batch, n_classes)
