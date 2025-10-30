import pennylane as qml
import torch
import numpy as np

class EstimatorQNN:
    """
    Quantum neural network using Pennylane variational circuit.
    Uses 3 qubits and 2 parameterized layers, suitable for hybrid training.
    """
    def __init__(self, num_qubits: int = 3, layers: int = 2):
        self.num_qubits = num_qubits
        self.layers = layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        # The circuit accepts a parameter matrix of shape (layers, num_qubits)
        # and an input vector of length num_qubits.
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, params: torch.Tensor, inputs: torch.Tensor):
        # Encode classical data into the first qubits via Ry rotations
        for i, w in enumerate(inputs):
            qml.RY(w, wires=i)
        # Parameterised layers with entanglement
        for l in range(self.layers):
            for i in range(self.num_qubits):
                qml.RY(params[l, i], wires=i)
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        # Measurement expectation of PauliZ on qubit 0
        return qml.expval(qml.PauliZ(0))

    def __call__(self, params: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the quantum circuit.
        """
        return self.qnode(params, inputs)

__all__ = ["EstimatorQNN"]
