"""Quantum self‑attention with a variational circuit that outputs a probability distribution
over the attention map.  The module retains the original `SelfAttention` API
but adds a depth‑controlled, weight‑sharing design that can be run either
on a simulator or a real device.

The circuit uses RX/RY/RZ rotations followed by controlled‑X entanglements,
mirroring the classical attention score calculation.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from typing import List, Dict


class SelfAttentionDual:
    """Quantum self‑attention with hybrid‑style depth control."""

    def __init__(self, n_qubits: int = 4, depth: int = 4, share_weights: bool = False):
        """
        Parameters
        ----------
        n_qubits : int, default=4
            Number of qubits per head (must match embed_dim in the classical branch).
        depth : int, default=4
            Number of attention heads.
        share_weights : bool, default=False
            Placeholder to match the classical API; not used in the quantum branch.
        """
        self.n_qubits = n_qubits
        self.depth = depth
        self.share_weights = share_weights
        # Random variational parameters for each head
        self.q_rot = np.random.rand(depth, n_qubits, 3)  # rx, ry, rz angles
        self.entanglement = np.random.rand(depth, n_qubits - 1)  # placeholder for future use

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        """Build a single‑head variational circuit."""
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)

        # rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # entanglement
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)

        circuit.measure_all()
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> List[Dict[str, int]]:
        """Execute the circuit and return the attention map as a list of
        measurement count dictionaries, one per head.
        """
        results: List[Dict[str, int]] = []
        for h in range(self.depth):
            circ = self._build_circuit(rotation_params, entangle_params)
            job = qiskit.execute(circ, backend, shots=shots)
            results.append(job.result().get_counts(circ))
        return results
