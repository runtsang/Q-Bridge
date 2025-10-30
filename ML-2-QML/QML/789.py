"""Quantum self‑attention block using a parameterised variational circuit.

The circuit implements a multi‑qubit rotation layer followed by
controlled‑rotation entanglement.  The class is compatible with
the classical interface: run(..., rotation_params, entangle_params)
returns measurement counts.  An optional method converts the
counts into a probability‑based attention matrix.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute


class SelfAttentionBlock:
    """
    Quantum self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Total number of qubits (should match embed_dim).
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        """Construct the parameterised circuit."""
        circuit = QuantumCircuit(self.qr, self.cr)

        # Rotation layer
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Entanglement layer (controlled‑RZ)
        for i in range(self.n_qubits - 1):
            circuit.crz(entangle_params[i], i, i + 1)

        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        """Execute the circuit on the supplied backend."""
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

    def attention_from_counts(self, counts: dict, seq_len: int) -> np.ndarray:
        """
        Convert measurement counts into a soft‑attention matrix.

        Parameters
        ----------
        counts : dict
            Measurement outcome counts from run().
        seq_len : int
            Sequence length (must be <= n_qubits).

        Returns
        -------
        weights : np.ndarray
            Attention matrix of shape (seq_len, seq_len).
        """
        # Map each bitstring to a probability
        probs = np.array([counts.get(format(i, f"0{self.n_qubits}b"), 0) for i in range(2**self.n_qubits)])
        probs = probs / probs.sum()

        # For simplicity, use the first seq_len qubits as query/key indices
        idx = np.arange(seq_len)
        weights = np.outer(probs[idx], probs[idx])
        return weights

    @staticmethod
    def default_backend():
        """Return a default Aer simulator."""
        return Aer.get_backend("qasm_simulator")
