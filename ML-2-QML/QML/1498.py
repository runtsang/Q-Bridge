"""Quantum fully connected layer with variational circuit and gradient support."""
import pennylane as qml
import numpy as np
from typing import Iterable

class FCL:
    def __init__(self, n_qubits: int = 1, device_name: str = "default.qubit", shots: int = 100):
        self.n_qubits = n_qubits
        self.dev = qml.device(device_name, wires=n_qubits, shots=shots)
        self.qnode = self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            # Parametrized rotations
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            # Entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            # Expectation of Z on all qubits
            exp = 0.0
            for i in range(self.n_qubits):
                exp += qml.expval(qml.PauliZ(i))
            return exp
        return circuit

    def forward(self, thetas: Iterable[float]) -> np.ndarray:
        """Run the variational circuit on input parameters ``thetas``."""
        return np.array([self.qnode(np.array(thetas, dtype=np.float64))])

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Compatibility wrapper: returns NumPy array of expectation values."""
        return self.forward(thetas)

    def gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """Compute the gradient of the expectation w.r.t. the input parameters."""
        grads = qml.grad(self.qnode)(np.array(thetas, dtype=np.float64))
        return grads
