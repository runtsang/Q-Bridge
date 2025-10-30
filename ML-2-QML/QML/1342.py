import pennylane as qml
import numpy as np
from typing import Iterable

class FCLGen299:
    """
    Variational quantum circuit implementing a fully connected layer.
    Provides a `run(thetas)` interface identical to the classical
    implementation, but uses entangling gates to capture quantum
    correlations.
    """

    def __init__(self, n_qubits: int = 1, dev_name: str = "default.qubit", shots: int = 1024):
        self.n_qubits = n_qubits
        self.dev = qml.device(dev_name, wires=n_qubits, shots=shots)

        @qml.qnode(self.dev)
        def circuit(params):
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the variational circuit with the supplied parameters.
        Returns the expectation value as a NumPy array, matching
        the classical run signature.
        """
        params = np.array(list(thetas), dtype=float)
        expectation = self.circuit(params)
        return np.array([expectation])

__all__ = ["FCLGen299"]
