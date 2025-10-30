import pennylane as qml
import numpy as np
from typing import Iterable, Sequence

class HybridFCL:
    """
    Variational quantum circuit that emulates a fully‑connected layer.
    Uses a parameter‑shiftable ansatz with entanglement and expectation
    value read‑out, providing a quantum analogue of the classical network.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        dev: qml.Device = None,
        shots: int = 1024
    ) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.dev = dev or qml.device("default.qubit", wires=n_qubits, shots=shots)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(thetas):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(thetas[i], wires=i)
            # Entangling layer
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measurement of PauliZ expectation on all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the circuit for a list of parameters and return the
        expectation values as a NumPy array.
        """
        thetas = np.array(thetas, dtype=float)
        return np.array(self.circuit(thetas))

def FCL() -> HybridFCL:
    """Compatibility wrapper returning a default instance."""
    return HybridFCL()

__all__ = ["HybridFCL", "FCL"]
