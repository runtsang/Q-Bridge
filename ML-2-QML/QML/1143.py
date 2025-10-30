import pennylane as qml
import numpy as np
from typing import Iterable

class FCL:
    """Variational quantum circuit mimicking a fully‑connected layer."""
    def __init__(self, n_qubits: int = 4, device: qml.Device | None = None,
                 shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.dev = device or qml.device("default.qubit", wires=n_qubits, shots=shots)
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            for w in range(self.n_qubits):
                qml.Hadamard(wires=w)
            for w, p in enumerate(params):
                qml.RY(p, wires=w)
            for w in range(self.n_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
            return qml.expval(qml.PauliZ(self.n_qubits - 1))
        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit and return expectation of last qubit."""
        params = np.array(list(thetas), dtype=np.float64)
        expectation = self.circuit(params)
        return np.array([expectation])

    def gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """Compute gradient of the expectation w.r.t. parameters."""
        params = np.array(list(thetas), dtype=np.float64)
        grad_fn = qml.grad(self.circuit)
        return grad_fn(params)

__all__ = ["FCL"]
