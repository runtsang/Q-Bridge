"""QuantumNAT Enhanced – quantum variational block using Pennylane."""

import pennylane as qml
import torch
import torch.nn as nn

__all__ = ["QuantumNAT_Enhanced"]

class QuantumNAT_Enhanced(nn.Module):
    """
    Quantum variational block that maps a 16-dim input vector to a 4-dim latent.
    Uses Pennylane's default qubit simulator.
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Device with batch support
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        # Trainable weights for each layer and qubit
        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3, dtype=torch.float64)
        )
        # Build QNode
        @qml.qnode(self.dev, interface="torch")
        def circuit(x, weights):
            # Encode input features into rotation angles
            for i in range(self.n_qubits):
                qml.RX(x[i], wires=i)
            # Variational layers
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)
                # Entangling layer
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Measure expectation values of Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of inputs.
        x: Tensor of shape (batch_size, n_qubits)
        Returns: Tensor of shape (batch_size, n_qubits)
        """
        if x.shape[1]!= self.n_qubits:
            raise ValueError(f"Expected input dim {self.n_qubits}, got {x.shape[1]}")
        return self.circuit(x, self.weights)

    def sample(self, x: torch.Tensor, n_shots: int = 1000) -> torch.Tensor:
        """
        Sample measurement outcomes from the quantum circuit.
        Returns: Tensor of shape (batch_size, n_qubits, n_shots)
        """
        @qml.qnode(self.dev, interface="torch", shots=n_shots)
        def sample_circuit(x, weights):
            for i in range(self.n_qubits):
                qml.RX(x[i], wires=i)
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.sample(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return sample_circuit(x, self.weights)
