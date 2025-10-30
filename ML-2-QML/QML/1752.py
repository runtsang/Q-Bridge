import pennylane as qml
import numpy as np
from typing import Iterable

class FCL:
    """
    A variational quantum circuit that emulates a fully connected layer.
    The circuit is built using Pennylane and consists of alternating
    rotation and entanglement layers. The expectation value of PauliZ
    on the last qubit is returned as the layer output.
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 2,
                 backend: str = "default.qubit", shots: int = 1024) -> None:
        self.dev = qml.device(backend, wires=n_qubits, shots=shots)
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params, thetas):
            # encode input parameters into rotations
            for i in range(self.n_qubits):
                qml.RY(thetas[i], wires=i)
            # variational layers
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(params[layer, i, 0], wires=i)
                    qml.RZ(params[layer, i, 1], wires=i)
                # entanglement
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            return qml.expval(qml.PauliZ(self.n_qubits - 1))

        # initialize trainable parameters
        self.params = np.random.uniform(-np.pi, np.pi,
                                        (self.n_layers, self.n_qubits, 2))
        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the circuit for a given set of input parameters.
        Thetas must be at least as long as the number of qubits.
        """
        # pad or truncate thetas to match number of qubits
        thetas_arr = np.array(thetas[:self.n_qubits], dtype=np.float32)
        if len(thetas_arr) < self.n_qubits:
            thetas_arr = np.pad(thetas_arr, (0, self.n_qubits - len(thetas_arr)))
        expectation = self.circuit(self.params, thetas_arr)
        return np.array([expectation])

__all__ = ["FCL"]
