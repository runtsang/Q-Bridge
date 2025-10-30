import pennylane as qml
import numpy as np
from typing import Iterable

class FCL:
    """
    Variational quantum circuit implementing a fully‑connected layer.
    Uses a layer of Y‑rotations, a chain of CNOT entanglements, and
    returns the expectation value of Pauli‑Z on the first qubit.
    """

    def __init__(self, n_qubits: int = 1, shots: int = 100, device: str = "default.qubit") -> None:
        self.n_qubits = n_qubits
        dev = qml.device(device, wires=n_qubits, shots=shots)

        @qml.qnode(dev, interface="autograd")
        def circuit(params):
            # Parameterised rotations
            for i in range(n_qubits):
                qml.RY(params[i], wires=i)
            # Entanglement
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measurement
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the circuit with the supplied parameters.
        `thetas` must contain at least `n_qubits` values; extra
        parameters are ignored, missing values are padded with zero.
        """
        thetas = np.asarray(list(thetas), dtype=np.float32)[:self.n_qubits]
        if thetas.size < self.n_qubits:
            thetas = np.pad(thetas, (0, self.n_qubits - thetas.size), mode="constant")
        result = self.circuit(thetas)
        return np.array([result])

__all__ = ["FCL"]
