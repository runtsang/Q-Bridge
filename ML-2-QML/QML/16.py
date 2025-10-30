"""Quantum self‑attention using Pennylane with multi‑head support."""

import pennylane as qml
import numpy as np


class QuantumSelfAttention:
    """
    Multi‑head quantum self‑attention block.
    Parameters
    ----------
    n_heads : int
        Number of attention heads (each head uses 4 qubits).
    """
    def __init__(self, n_heads: int = 4):
        self.n_heads = n_heads
        self.n_qubits = 4 * n_heads
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        @qml.qnode(self.dev, interface="auto")
        def circuit():
            # Apply rotations per head
            for h in range(self.n_heads):
                qubits = range(4 * h, 4 * h + 4)
                for i, q in enumerate(qubits):
                    qml.RX(rotation_params[q, 0], wires=q)
                    qml.RY(rotation_params[q, 1], wires=q)
                    qml.RZ(rotation_params[q, 2], wires=q)

            # Entangling across heads
            for h in range(self.n_heads - 1):
                qml.CRX(entangle_params[h], wires=[4 * h + 3, 4 * (h + 1)])

            # Measure in computational basis
            return qml.probs(wires=range(self.n_qubits))

        return circuit()

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        """
        Execute the attention circuit.
        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (n_qubits, 3) with RX,RY,RZ angles per qubit.
        entangle_params : np.ndarray
            Shape (n_heads-1,) with CRX angles between heads.
        shots : int
            Number of shots for measurement (ignored in default.qubit simulator).
        Returns
        -------
        dict
            Probability distribution over computational basis states.
        """
        probs = self._build_circuit(rotation_params, entangle_params)
        # Convert probabilities to counts for consistency with Qiskit output
        counts = {
            bin(i)[2:].zfill(self.n_qubits): round(p * shots, 5)
            for i, p in enumerate(probs)
        }
        return counts


def SelfAttention():
    """
    Factory function mirroring the original API.
    Returns an instance of QuantumSelfAttention with default parameters.
    """
    return QuantumSelfAttention(n_heads=4)


__all__ = ["SelfAttention"]
