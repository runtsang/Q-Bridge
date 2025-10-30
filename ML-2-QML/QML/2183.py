import pennylane as qml
import numpy as np
from typing import Iterable

class FCL:
    """
    Parameterized quantum circuit with entanglement and shot‑based expectation.
    The ``run`` method evaluates the circuit for a batch of parameter vectors.
    """
    def __init__(self, n_qubits: int = 4, device_name: str = "default.qubit", shots: int = 1000):
        self.n_qubits = n_qubits
        self.device = qml.device(device_name, wires=n_qubits, shots=shots)
        self._circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.device, interface="autograd")
        def circuit(params):
            # Parametrized rotation on each qubit
            for w in range(self.n_qubits):
                qml.Hadamard(wires=w)
            for w, theta in enumerate(params):
                qml.RY(theta, wires=w)
            # Entangle via a CNOT chain
            for w in range(self.n_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
            # Expectation of Pauli‑Z on the first qubit
            return qml.expval(qml.PauliZ(0))
        return circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluates the circuit for a single parameter vector.
        Expects ``thetas`` to have length equal to ``n_qubits``.
        """
        params = np.array(list(thetas), dtype=np.float64)
        if params.size!= self.n_qubits:
            raise ValueError(f"Input theta must have length {self.n_qubits}")
        expectation = self._circuit(params)
        return np.array([expectation])

    def gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Computes the gradient of the expectation value w.r.t. the parameters
        using Pennylane's automatic differentiation.
        """
        params = np.array(list(thetas), dtype=np.float64)
        if params.size!= self.n_qubits:
            raise ValueError(f"Input theta must have length {self.n_qubits}")
        grad = qml.grad(self._circuit)(params)
        return grad

__all__ = ["FCL"]
