"""Quantum self‑attention using a Pennylane variational circuit.

The circuit implements a parameterized rotation layer followed by
controlled‑phase entanglement.  The output is a probability distribution
over qubit indices that serves as attention logits.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

class QuantumSelfAttention:
    """Hybrid variational circuit producing attention logits."""

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(rotation_params, entangle_params):
            # Apply single‑qubit rotations
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)

            # Entangling layer: controlled‑phase gates
            for i in range(self.n_qubits - 1):
                qml.CPHASE(entangle_params[i], wires=[i, i + 1])

            # Measure Z expectation on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the variational circuit and return a softmaxed attention
        distribution derived from the absolute Z‑expectations.

        Parameters
        ----------
        rotation_params : np.ndarray
            Array of shape (3 * n_qubits,) specifying RX, RY, RZ angles.
        entangle_params : np.ndarray
            Array of shape (n_qubits - 1,) specifying CPHASE angles.
        shots : int, optional
            Number of simulation shots (unused for statevector but kept for API consistency).

        Returns
        -------
        np.ndarray
            Attention logits of shape (n_qubits,).
        """
        # Run the circuit
        expvals = self.circuit(rotation_params, entangle_params)
        # Convert to probabilities
        logits = np.abs(expvals)  # magnitude of Z expectation as a proxy
        probs = logits / logits.sum() if logits.sum()!= 0 else np.ones_like(logits) / len(logits)
        return probs

def SelfAttention() -> QuantumSelfAttention:
    """Return a quantum attention object configured for 4 qubits."""
    return QuantumSelfAttention(n_qubits=4)

__all__ = ["SelfAttention"]
