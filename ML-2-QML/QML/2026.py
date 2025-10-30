import pennylane as qml
import numpy as np
from typing import Iterable

class QuantumFullyConnectedLayer:
    """
    Variational quantum circuit that emulates a fully‑connected layer.
    The circuit consists of a single qubit layer with a parameterized
    Ry rotation followed by a measurement of the Z observable.  The
    design mirrors the classical surrogate but uses Pennylane’s
    quantum simulator, allowing easy extension to multi‑qubit topologies
    or different ansätze.
    """
    def __init__(self, n_qubits: int = 1, backend: str = "default.qubit",
                 shots: int = 1000) -> None:
        self.n_qubits = n_qubits
        self.device = qml.device(backend, wires=n_qubits, shots=shots)
        self._theta = qml.param("theta")

        @qml.qnode(self.device, interface="autograd")
        def circuit(theta):
            qml.Hadamard(wires=range(n_qubits))
            for i in range(n_qubits):
                qml.RY(theta[i], wires=i)
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the variational circuit with the provided parameters.
        * ``thetas`` must be a list of length *n_qubits*.
        * Returns a NumPy array containing the expectation value of
          the Z operator on the first qubit, matching the original
          API.
        """
        theta_arr = np.array(list(thetas), dtype=np.float64)
        if theta_arr.size!= self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} parameters, got {theta_arr.size}")
        exp_val = self.circuit(theta_arr)
        return np.array([exp_val])

__all__ = ["QuantumFullyConnectedLayer"]
