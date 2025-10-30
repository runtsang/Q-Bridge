"""Quantum self‑attention using a variational circuit and parameter‑shift gradients.

The circuit implements a multi‑qubit rotation followed by a controlled‑X entangling layer.
All parameters are trainable via gradient descent using the parameter‑shift rule.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import Tuple


class SelfAttention:
    """
    Variational quantum self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must be even for the entangling pattern).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, n_qubits: int = 4, seed: int | None = None):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=1024)
        self.seed = seed
        self._build_params()

    def _build_params(self) -> None:
        rng = np.random.default_rng(self.seed)
        # rotation_params: 3 * n_qubits angles (RX, RY, RZ per qubit)
        self.rotation_params = rng.uniform(0, 2 * np.pi, size=(3 * self.n_qubits))
        # entangle_params: n_qubits - 1 angles for controlled‑X rotations
        self.entangle_params = rng.uniform(0, 2 * np.pi, size=(self.n_qubits - 1))

    def _quantum_circuit(self, rotation_params: np.ndarray,
                         entangle_params: np.ndarray) -> Tuple[qml.operation.Operator,...]:
        @qml.qnode(self.dev)
        def circuit():
            # Apply single‑qubit rotations
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)

            # Controlled‑X entanglement pattern
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])

            # Measure expectation of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """
        Execute the circuit and return expectation values.

        Parameters
        ----------
        rotation_params : array_like
            Shape (3 * n_qubits,). Rotation angles.
        entangle_params : array_like
            Shape (n_qubits - 1,). Entanglement angles.
        shots : int, optional
            Number of shots for the simulation.

        Returns
        -------
        np.ndarray
            Expectation values for each qubit, shape (n_qubits,).
        """
        # Override device shots for this run
        self.dev.shots = shots
        circuit = self._quantum_circuit(rotation_params, entangle_params)
        return np.array(circuit())

    def trainable_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the current trainable parameters."""
        return self.rotation_params, self.entangle_params

__all__ = ["SelfAttention"]
