"""QuantumHybridClassifier – quantum‑classical hybrid network with advanced quantum integration.

This module extends the original hybrid architecture by:
- Using a parameter‑shift rule for analytic gradients.
- Supporting multiple quantum backends (Pennylane default, Aer, or a custom device).
- Providing a configurable number of output classes.
- Adding a small classical post‑processing head after the quantum expectation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pennylane as qml
from pennylane import numpy as pnp

__all__ = ["QuantumHybridClassifier"]

class VariationalQuantumCircuit(nn.Module):
    """
    A parameterised two‑qubit variational circuit that returns the expectation of Z⊗Z.
    The circuit supports analytic gradients via Pennylane's automatic differentiation.
    """
    def __init__(self,
                 n_qubits: int = 2,
                 layers: int = 2,
                 device: str = "default.qubit") -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.layers = layers
        self.device = device
        self.dev = qml.device(device, wires=n_qubits)

        # Create a quantum node (qnode) that returns the expectation value
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            # Apply a rotation on the first qubit that encodes the classical input
            qml.RX(x, wires=0)
            for l in range(layers):
                for w in range(n_qubits):
                    qml.RX(params[l, w, 0], wires=w)
                    qml.RY(params[l, w, 1], wires=w)
                    qml.RZ(params[l, w, 2], wires=w)
                # Entangling layer
                for w in range(n_qubits - 1):
                    qml.CNOT(wires=[w, w + 1])
            return qml.expval(qml.Z(0) @ qml.Z(1))

        self.circuit = circuit
        # Initialise parameters
        init_params = pnp.random.uniform(low=0, high=2 * np.pi,
                                         size=(layers, n_qubits, 3))
        self.params = nn.Parameter(torch.tensor(init_params, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute the expectation value for each element of x.
        x is expected to be a 1‑D tensor of shape (batch,) or (batch, 1).
        """
        # Ensure x is a 1‑D tensor
        x = x.view(-1)
        # Compute expectation values
        out = self.circuit(self.params, x)
        return out.view(-1, 1)

class HybridLayer(nn.Module):
    """
    Hybrid layer that forwards activations through a variational quantum circuit.
    """
    def __init__(self,
                 n_qubits: int = 2,
                 layers: int = 2,
                 device: str = "default.qubit") -> None:
        super().__init__()
        self.quantum_circuit = VariationalQuantumCircuit(n_qubits, layers, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x shape (batch, 1)
        return self.quantum_circuit(x)

class QuantumHybridClassifier(nn.Module):
    """
    Convolutional network followed by a variational quantum head and a classical
    post‑processing layer that outputs class probabilities.
    """
    def __init__(self,
                 n_qubits: int = 2,
                 layers: int = 2,
                 device: str = "default.qubit",
                 num_classes: int = 2) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully‑connected head
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Hybrid quantum layer
        self.hybrid = HybridLayer(n_qubits=n_qubits, layers=layers, device=device)

        # Classical post‑processing head
        self.post_head = nn.Linear(1, num_classes)

        # Activation
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional backbone
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)

        # Flatten and fully‑connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Quantum hybrid layer
        q_out = self.hybrid(x)

        # Classical post‑processing
        logits = self.post_head(q_out)
        probs = self.softmax(logits)

        return probs
