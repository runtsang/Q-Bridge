"""Quantum self‑attention implemented with Pennylane.

The circuit applies RX/RZ rotations per qubit, a chain of CRX entanglement,
and reads out the expectation values of Pauli‑Z on each qubit.  The
`run` method returns a NumPy array of measurement values, preserving the
original seed's interface.
"""

import numpy as np
import pennylane as qml

class QuantumSelfAttention:
    """Variational circuit mimicking a self‑attention block."""
    def __init__(self, n_qubits: int = 4, wires: list | None = None):
        self.n_qubits = n_qubits
        self.wires = wires or list(range(n_qubits))
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="numpy")
        def circuit(rotation_params: np.ndarray, entangle_params: np.ndarray):
            # Encode inputs as rotations
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=self.wires[i])
                qml.RY(rotation_params[3 * i + 1], wires=self.wires[i])
                qml.RZ(rotation_params[3 * i + 2], wires=self.wires[i])

            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[self.wires[i], self.wires[i + 1]])

            # Expectation values of Pauli‑Z
            return [qml.expval(qml.PauliZ(w)) for w in self.wires]
        self.circuit = circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """Execute the circuit and return expectation values."""
        return self.circuit(rotation_params, entangle_params)

def SelfAttention():
    """Factory returning a ready‑to‑use instance."""
    return QuantumSelfAttention(n_qubits=4)

__all__ = ["SelfAttention"]
