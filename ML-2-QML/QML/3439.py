"""Quantum self‑attention circuit inspired by the QML seed and the classifier depth logic.

The ansatz encodes the input embedding into rotation gates, then applies a
sequence of variational Ry rotations and CZ entangling layers.  After measurement
each qubit yields a bit‑string that is interpreted as a probability distribution
over the attention indices.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.providers import Backend
from typing import Dict

class QuantumSelfAttentionEnhanced:
    """Variational quantum circuit that outputs a probability distribution
    over the attention dimension."""

    def __init__(self, n_qubits: int, depth: int = 1) -> None:
        self.n_qubits = n_qubits
        self.depth = depth

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        """Construct the parameterised circuit."""
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)

        # Encoding
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Variational layers
        for _ in range(self.depth):
            for i in range(self.n_qubits):
                circuit.ry(entangle_params[i % len(entangle_params)], i)
            for i in range(self.n_qubits - 1):
                circuit.cz(i, i + 1)

        circuit.measure(qr, cr)
        return circuit

    def run(
        self,
        backend: Backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> Dict[str, int]:
        """Execute the circuit and return the raw bit‑string counts."""
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

    def state_dict(self) -> Dict[str, int]:
        """State dictionary for checkpointing."""
        return {"n_qubits": self.n_qubits, "depth": self.depth}

__all__ = ["QuantumSelfAttentionEnhanced"]
