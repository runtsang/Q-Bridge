import pennylane as qml
import torch
from pennylane import numpy as np

class QCNNModel:
    """Quantum QCNN implemented with Pennylane variational layers."""
    def __init__(self, n_qubits: int = 8, n_layers: int = 3, device: qml.Device | None = None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = device or qml.device("default.qubit", wires=n_qubits)
        # Weight tensor: (layer, qubit, rotation_type)
        self.weights = torch.nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        self.feature_map = qml.templates.FeaturesMap.ZFeatureMap(n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: torch.Tensor) -> torch.Tensor:
        # Encode classical data
        self.feature_map(x)
        # Variational ansatz: alternating rotations and CZ entanglement
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                qml.RZ(self.weights[layer, q, 0], wires=q)
                qml.RY(self.weights[layer, q, 1], wires=q)
            for q in range(self.n_qubits - 1):
                qml.CZ(wires=[q, q + 1])
            for q in range(self.n_qubits):
                qml.RZ(self.weights[layer, q, 2], wires=q)
        # Measurement of Pauli Z on the first qubit
        return qml.expval(qml.PauliZ(0))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return a probability by applying a sigmoid to the circuit expectation value.
        """
        return torch.sigmoid(self.qnode(x))
