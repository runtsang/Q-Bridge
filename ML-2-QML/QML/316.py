import pennylane as qml
import numpy as np
from typing import Iterable

class FCL:
    """
    Variational quantum circuit that mimics a fully‑connected layer.  The
    parameter vector *thetas* is interpreted as a flattened list of rotation
    angles for a chain of Ry/Rz gates followed by a layer of CNOT entanglement.
    The output is the expectation value of Z on the first qubit.
    """

    def __init__(self, n_qubits: int = 4, layers: int = 2, backend: str = "default.qubit",
                 shots: int = 1024):
        self.n_qubits = n_qubits
        self.layers = layers
        self.shots = shots
        self.dev = qml.device(backend, wires=n_qubits, shots=shots)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params: np.ndarray):
            for l in range(layers):
                for q in range(n_qubits):
                    qml.RY(params[l, q, 0], wires=q)
                    qml.RZ(params[l, q, 1], wires=q)
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the circuit with a flattened parameter vector.  The vector is
        reshaped to (layers, n_qubits, 2) before being fed to the circuit.
        """
        param_shape = (self.layers, self.n_qubits, 2)
        params = np.array(list(thetas)).reshape(param_shape)
        return np.array([self._circuit(params)])

    def gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Return the gradient of the expectation value with respect to thetas.
        """
        param_shape = (self.layers, self.n_qubits, 2)
        params = np.array(list(thetas)).reshape(param_shape)
        return np.array([qml.grad(self._circuit)(params)])

    def count_params(self) -> int:
        return self.layers * self.n_qubits * 2

__all__ = ["FCL"]
