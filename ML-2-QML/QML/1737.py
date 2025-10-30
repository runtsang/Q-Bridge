"""
QCNNGen323QML: A Pennylane‑based hybrid quantum–classical circuit.

Features:
* ZFeatureMap for data encoding.
* Entanglement layer with tunable depth.
* Parameterised rotations as the ansatz.
* Explicit cost function and simple gradient descent training loop.
"""

import pennylane as qml
import numpy as np
import torch
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer


class QCNNGen323QML:
    def __init__(self, num_qubits: int = 8, depth: int = 3, lr: float = 0.01):
        self.num_qubits = num_qubits
        self.depth = depth
        self.dev = qml.device("default.qubit", wires=num_qubits)

        # Feature map
        self.feature_map = qml.templates.feature_maps.ZFeatureMap(num_qubits)

        # Trainable parameters
        self.weights = np.random.randn(depth, num_qubits, 3)  # rotations around X,Y,Z

        # Define the QNode
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: np.ndarray):
            # Encode data
            self.feature_map(inputs)
            # Ansatz with entanglement
            for layer in range(depth):
                for qubit in range(num_qubits):
                    qml.RX(weights[layer, qubit, 0], wires=qubit)
                    qml.RY(weights[layer, qubit, 1], wires=qubit)
                    qml.RZ(weights[layer, qubit, 2], wires=qubit)
                # Entangle neighbours (cyclic)
                for q in range(num_qubits):
                    qml.CNOT(wires=[q, (q + 1) % num_qubits])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        self.optimizer = AdamOptimizer(stepsize=lr)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the expectation value for given classical inputs."""
        return self.circuit(inputs, self.weights)

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Binary cross‑entropy loss."""
        return torch.nn.functional.binary_cross_entropy(outputs, targets)

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """One training step using Adam optimizer."""
        loss_value = self.loss(self.forward(inputs), targets)
        self.optimizer.step(lambda w: self.loss(self.circuit(inputs, w), targets), self.weights)
        return loss_value

    def train(self, data_loader, epochs: int = 50):
        """Full training loop over a PyTorch DataLoader."""
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_inputs, batch_targets in data_loader:
                loss = self.train_step(batch_inputs, batch_targets)
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} – Loss: {epoch_loss / len(data_loader):.4f}")

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return probability predictions."""
        return torch.sigmoid(self.forward(inputs))

__all__ = ["QCNNGen323QML"]
