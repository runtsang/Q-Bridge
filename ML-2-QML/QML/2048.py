"""Variational quantum fully‑connected layer using Pennylane.
The circuit supports up to `n_qubits` qubits, applies parameterized
RY rotations followed by a chain of CNOT gates for entanglement,
and measures the expectation value of Pauli‑Z on the first qubit.
The interface matches the classical `run` method, returning a NumPy array."""
from __future__ import annotations

import numpy as np
import pennylane as qml


class FCLLayer:
    """
    Quantum implementation of a fully‑connected layer. Parameters are
    interpreted as rotation angles on individual qubits. An entangling
    layer ensures correlations between qubits, allowing the circuit to
    emulate nonlinear classical transformations.
    """

    def __init__(self, n_qubits: int = 4, dev: qml.Device | None = None, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.dev = dev or qml.device("default.qubit", wires=n_qubits, shots=shots)
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> qml.QNode:
        @qml.qnode(self.dev)
        def circuit(params: np.ndarray) -> float:
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))
        return circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit with the supplied parameters and return the expectation."""
        params = np.array(list(thetas), dtype=np.float32)
        if len(params)!= self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} parameters, got {len(params)}.")
        expectation = self._circuit(params)
        return np.array([expectation])


def FCL() -> FCLLayer:
    """Convenience factory matching the original API."""
    return FCLLayer()


__all__ = ["FCLLayer", "FCL"]
