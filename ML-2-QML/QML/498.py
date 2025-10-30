"""Quantum fully connected layer using Pennylane with a variational ansatz."""
import numpy as np
import pennylane as qml
from typing import Iterable

class FullyConnectedLayerGen137:
    """A parameterized quantum circuit that emulates a fully connected layer.
    The circuit consists of a layer of rotation gates followed by a
    variational entangling block. Expectation values are computed
    with respect to the Z observable."""
    def __init__(self, n_qubits: int = 1, device_name: str = "default.qubit", shots: int = 1000):
        self.n_qubits = n_qubits
        self.device = qml.device(device_name, wires=n_qubits, shots=shots)

        @qml.qnode(self.device, interface="numpy")
        def circuit(params):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(params[i], wires=i)
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Evaluate the circuit for a batch of parameter vectors."""
        thetas = np.array(thetas, dtype=np.float32)
        if thetas.ndim == 1:
            thetas = thetas.reshape(1, -1)
        expectations = []
        for params in thetas:
            exp = self._circuit(params)
            expectations.append(exp)
        return np.array(expectations)

def FCL() -> FullyConnectedLayerGen137:
    """Factory that returns an instance of the quantum fully connected layer."""
    return FullyConnectedLayerGen137(n_qubits=1, shots=1024)

__all__ = ["FCL", "FullyConnectedLayerGen137"]
