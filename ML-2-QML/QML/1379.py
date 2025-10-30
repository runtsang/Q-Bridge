import pennylane as qml
from pennylane import numpy as np

class EstimatorQNNAdvanced:
    """Variational quantum circuit for regression with entanglement and multi‑observable readout.
    Matches the classical interface but operates on qubit states using Pennylane.
    The circuit uses an RY encoding, a stack of parameterised rotations, and
    a simple entangling CNOT ladder. Observables are Pauli‑Z on each qubit,
    producing a vector that can be linearly combined to predict a scalar."""
    def __init__(self, num_qubits: int = 2, num_layers: int = 3):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.device = qml.device("default.qubit", wires=num_qubits)
        self.params = np.random.randn(num_layers, num_qubits, 3)
        self.observables = [qml.PauliZ(i) for i in range(num_qubits)]

    def circuit(self, inputs: np.ndarray, weights: np.ndarray):
        for i in range(self.num_qubits):
            qml.RY(inputs[i], wires=i)
        for layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                qml.Rot(*weights[layer, qubit], wires=qubit)
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return [qml.expval(obs) for obs in self.observables]

    def __call__(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return self.circuit(inputs, weights)

__all__ = ["EstimatorQNNAdvanced"]
