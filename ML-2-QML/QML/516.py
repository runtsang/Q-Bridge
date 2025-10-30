import pennylane as qml
import numpy as np
from typing import Iterable

__all__ = ["FCL"]

class FCL:
    """
    Variational quantum circuit that mimics a fully‑connected layer.
    The circuit applies a Hadamard to each qubit, a parameterised RY
    rotation, entangles adjacent qubits with CNOTs, and measures the
    expectation value of Pauli‑Z on the first qubit.
    """

    def __init__(self, n_qubits: int = 1, shots: int = 100, device=None):
        if device is None:
            device = qml.device("default.qubit", wires=n_qubits, shots=shots)
        self.device = device
        self.n_qubits = n_qubits

        @qml.qnode(self.device, interface="autograd")
        def circuit(thetas):
            # Prepare superposition
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            # Parameterised rotations
            for i, theta in enumerate(thetas):
                qml.RY(theta, wires=i)
            # Entangle neighbouring qubits
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measure expectation of Pauli‑Z on qubit 0
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Must contain `n_qubits` parameters.

        Returns
        -------
        np.ndarray
            The expectation value of Pauli‑Z on qubit 0 as a 1‑D array.
        """
        thetas = np.asarray(thetas, dtype=np.float64)
        if len(thetas)!= self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} parameters, got {len(thetas)}")
        expectation = self.circuit(thetas)
        return np.array([expectation])
