"""Quantum variational fully connected layer using Pennylane."""

import pennylane as qml
import numpy as np
from typing import Iterable

class FCL:
    """
    Parameterized quantum circuit that emulates a classical fully‑connected layer.
    The circuit consists of a layer of RY rotations (parameterized by ``thetas``),
    followed by a chain of CNOTs that entangles all qubits, and finally a
    measurement of the PauliZ expectation value on each qubit.  The returned
    vector contains the expectation value for each qubit, which can be
    interpreted as the output of a fully‑connected layer with one output.
    """
    def __init__(self, n_qubits: int = 1, dev: qml.Device = None) -> None:
        self.n_qubits = n_qubits
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev)
        def circuit(params: np.ndarray):
            # Apply a layer of RY rotations
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            # Entangle with a simple chain of CNOTs
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measure expectation value of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self._circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied parameters.
        Args:
            thetas: Iterable of rotation angles, one per qubit.
        Returns:
            A numpy array containing the expectation value of PauliZ on each qubit.
        """
        params = np.array(list(thetas), dtype=np.float32)
        if params.shape[0]!= self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} parameters, got {len(thetas)}.")
        return np.array(self._circuit(params))

__all__ = ["FCL"]
