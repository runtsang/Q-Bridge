"""Quantum self‑attention using Pennylane variational circuits."""
from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

class QuantumSelfAttentionEnhanced:
    """Variational circuit that emulates a self‑attention block.

    The circuit consists of a layer of single‑qubit rotations followed by a
    nearest‑neighbour entangling block.  The output is a probability vector
    over the computational basis of the first ``n_qubits`` qubits, which
    can be interpreted as attention scores.
    """
    def __init__(self, n_qubits: int, wires: list[int] | None = None):
        self.n_qubits = n_qubits
        self.wires = wires if wires is not None else list(range(n_qubits))
        self.dev = qml.device("default.qubit", wires=self.wires)

        @qml.qnode(self.dev, interface="autograd", diff_method="backprop")
        def circuit(rotation_params: np.ndarray,
                    entangle_params: np.ndarray):
            """Variational circuit."""
            # Single‑qubit rotations
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=self.wires[i])
                qml.RY(rotation_params[3 * i + 1], wires=self.wires[i])
                qml.RZ(rotation_params[3 * i + 2], wires=self.wires[i])

            # Entangling layer (nearest‑neighbour CNOTs with tunable rotations)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[self.wires[i], self.wires[i + 1]])
                qml.RZ(entangle_params[i], wires=self.wires[i + 1])

            # Expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(w)) for w in self.wires]

        self.circuit = circuit

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """Execute the variational circuit and return a probability‑like vector.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape ``(3 * n_qubits,)`` – rotation angles for RX, RY, RZ.
        entangle_params : np.ndarray
            Shape ``(n_qubits - 1,)`` – angles for the Z‑rotations after each CNOT.
        shots : int, optional
            Number of Monte‑Carlo shots for the simulator (ignored in the
            autograd device but kept for API compatibility).

        Returns
        -------
        np.ndarray
            Shape ``(n_qubits,)`` – expectation values of Pauli‑Z, scaled to
            ``[0, 1]`` to resemble attention weights.
        """
        # Ensure correct shapes
        rotation_params = np.asarray(rotation_params, dtype=np.float64)
        entangle_params = np.asarray(entangle_params, dtype=np.float64)

        raw = self.circuit(rotation_params, entangle_params)
        # Map from [-1, 1] to [0, 1]
        probs = 0.5 * (np.array(raw) + 1.0)
        return probs

__all__ = ["QuantumSelfAttentionEnhanced"]
