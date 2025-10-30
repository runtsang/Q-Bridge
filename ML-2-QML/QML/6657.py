import pennylane as qml
import torch
from torch import nn

class QuantumEncoder(nn.Module):
    """Variational quantum encoder using Pennylane's Torch interface."""
    def __init__(
        self,
        num_qubits: int,
        num_layers: int,
        depth: int = 1,
        device: str = "default.qubit",
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.depth = depth
        self.dev = qml.device(device, wires=num_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor, weights: torch.Tensor):
            # Input encoding: rotate each qubit by the corresponding feature
            for i in range(num_qubits):
                qml.RX(x[i], wires=i)
            # Variational layers
            for _ in range(depth):
                for layer in range(num_layers):
                    # Parameterised single‑qubit rotations
                    qml.RY(weights[layer, 0], wires=layer)
                    qml.RZ(weights[layer, 1], wires=layer)
                    # Entangling CNOT chain
                    for q in range(num_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])
            # Expectation value of PauliZ on first qubit
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        # Initialise weights
        self.weights = nn.Parameter(
            torch.randn(num_layers, 2, requires_grad=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum circuit.
        :param x: tensor of shape (num_qubits,)
        :return: tensor of shape (1,) containing the expectation value.
        """
        return self.circuit(x, self.weights)

def create_quantum_encoder(num_qubits: int, num_layers: int, depth: int = 1) -> QuantumEncoder:
    """Convenience constructor for QuantumEncoder."""
    return QuantumEncoder(num_qubits=num_qubits, num_layers=num_layers, depth=depth)
