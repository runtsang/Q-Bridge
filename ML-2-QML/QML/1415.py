import pennylane as qml
import numpy as np
from typing import Iterable, Union

class FCL:
    """
    Variational quantum circuit acting as a fully‑connected layer.
    Supports arbitrary number of qubits and batched evaluation.
    Uses Pennylane's autograd interface for easy integration with
    classical optimizers.
    """
    def __init__(self, n_qubits: int = 1, dev: qml.Device = None) -> None:
        self.n_qubits = n_qubits
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(theta):
            # Simple entangling layer: H on all qubits, then Ry
            qml.Hadamard(wires=range(n_qubits))
            qml.RY(theta, wires=range(n_qubits))
            # Measurement in Z basis
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def run(self, thetas: Union[Iterable[float], np.ndarray, list]) -> np.ndarray:
        """
        Evaluate the circuit for a batch of parameters.
        `thetas` can be a 1‑D array (single sample) or 2‑D (batch).
        Returns an array of expectation values.
        """
        thetas = np.array(thetas, dtype=np.float32)
        if thetas.ndim == 1:
            thetas = thetas.reshape(1, -1)
        expectations = []
        for theta in thetas:
            expectations.append(self.circuit(theta))
        return np.array(expectations)

__all__ = ["FCL"]
