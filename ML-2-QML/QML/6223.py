"""Quantum self‑attention module leveraging PennyLane."""

import pennylane as qml
import numpy as np

class QuantumSelfAttentionModule:
    """
    Variational quantum circuit that produces a similarity matrix between tokens.
    The circuit depth and gate parameters are configurable, enabling integration
    with a classical self‑attention block.
    """

    def __init__(self, n_qubits: int, depth: int = 2, device=None):
        self.n_qubits = n_qubits
        self.depth = depth
        self.device = device or qml.device("default.qubit", wires=n_qubits)

    def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        """Define the variational circuit and return a QNode."""

        @qml.qnode(self.device, interface="numpy")
        def circuit():
            # Apply single‑qubit rotations
            for i in range(self.n_qubits):
                idx = 3 * i
                qml.RX(rotation_params[idx], wires=i)
                qml.RY(rotation_params[idx + 1], wires=i)
                qml.RZ(rotation_params[idx + 2], wires=i)

            # Entangling layers
            for d in range(self.depth):
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(self.n_qubits):
                    ent_idx = d * self.n_qubits + i
                    qml.RZ(entangle_params[ent_idx], wires=i)

            # Measure pairwise ZZ correlations as similarity indicators
            expvals = []
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    expvals.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)))
            return np.array(expvals)

        return circuit()

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> np.ndarray:
        """
        Execute the circuit and return a symmetric similarity matrix
        of shape (n_qubits, n_qubits) with values in [0, 1].
        """
        raw = self._circuit(rotation_params, entangle_params)

        # Convert vector of pairwise correlations to matrix
        mat = np.zeros((self.n_qubits, self.n_qubits))
        idx = 0
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                mat[i, j] = mat[j, i] = (raw[idx] + 1) / 2  # map from [-1, 1] to [0, 1]
                idx += 1
        np.fill_diagonal(mat, 1.0)
        return mat

__all__ = ["QuantumSelfAttentionModule"]
