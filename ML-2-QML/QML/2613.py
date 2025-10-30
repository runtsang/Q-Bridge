"""Quantum self‑attention module built with Qiskit.

This class implements a variational self‑attention circuit that
mirrors the classical interface.  Rotation parameters control single‑qubit
rotations, while entanglement parameters control cross‑qubit
controlled‑X rotations.  The circuit is executed on a chosen backend
and returns a probability distribution over the computational basis,
which can be interpreted as an attention‑like similarity score.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from typing import Dict

class SelfAttentionFusion:
    """Quantum self‑attention using a variational circuit."""

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        """Construct the attention circuit from the provided parameters."""
        circuit = QuantumCircuit(self.qr, self.cr)
        # Apply per‑qubit rotations
        for i in range(self.n_qubits):
            rx, ry, rz = rotation_params[3 * i : 3 * i + 3]
            circuit.rx(rx, i)
            circuit.ry(ry, i)
            circuit.rz(rz, i)
        # Apply entangling gates
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> Dict[str, int]:
        """Execute the attention circuit and return measurement counts."""
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend=backend, shots=shots)
        return job.result().get_counts(circuit)

__all__ = ["SelfAttentionFusion"]
