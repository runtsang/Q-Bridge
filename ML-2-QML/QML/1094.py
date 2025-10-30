"""Quantum self‑attention using Pennylane variational circuits.

The circuit learns attention‑like weights via parameterised rotations
and entanglement layers.  The result is a vector of Pauli‑Z expectation
values that can be interpreted as attention scores.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def SelfAttention():
    class QuantumSelfAttention:
        """Variational quantum circuit that outputs attention‑style scores."""

        def __init__(self, n_qubits: int = 4, n_layers: int = 2):
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.dev = qml.device("default.qubit", wires=n_qubits)

            # Parameters will be created lazily in the circuit
            self.params = None

        def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
            """Variational circuit with rotation and entanglement layers."""
            # Ensure parameter shapes
            expected_rot = self.n_qubits * 3
            expected_ent = self.n_qubits * (self.n_layers - 1)
            if rotation_params.size!= expected_rot:
                raise ValueError(f"Expected {expected_rot} rotation params, got {rotation_params.size}")
            if entangle_params.size!= expected_ent:
                raise ValueError(f"Expected {expected_ent} entangle params, got {entangle_params.size}")

            # Rotation layer
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)

            # Entanglement layers
            idx = 0
            for _ in range(self.n_layers):
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(self.n_qubits - 1):
                    qml.CRX(entangle_params[idx], wires=[i, i + 1])
                    idx += 1

            # Measure expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        def run(
            self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024,
        ) -> np.ndarray:
            """Execute the circuit and return expectation values."""
            circuit = qml.QNode(self._circuit, self.dev)
            return circuit(rotation_params, entangle_params)

    # Default instance mirrors the 4‑qubit example
    return QuantumSelfAttention(n_qubits=4, n_layers=2)
