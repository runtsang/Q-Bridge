"""Quantum Self‑Attention hybrid module.

This module builds a parameterised quantum circuit that emulates a
self‑attention block.  The circuit consists of single‑qubit rotations
parameterised by ``rotation_params`` and two‑qubit CRX gates
parameterised by ``entangle_params``.  The output is a set of
attention weights derived from the expectation value of the Z operator
on each qubit.  These weights are used to compute a weighted sum of
the input features.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute


class SelfAttentionHybrid:
    """
    Quantum self‑attention block.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the key‑value space.  Must equal ``n_qubits``.
    n_qubits : int, optional
        Number of qubits used to encode the attention weights.
    shots : int, optional
        Number of shots for the simulator.
    """

    def __init__(self, embed_dim: int, n_qubits: int = 4, shots: int = 1024):
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.shots = shots

        if embed_dim!= n_qubits:
            raise ValueError("For the quantum implementation, embed_dim must equal n_qubits.")

        self.qr = QuantumRegister(n_qubits, name="q")
        self.cr = ClassicalRegister(n_qubits, name="c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        """Build the parameterised circuit for a single attention instance."""
        circuit = QuantumCircuit(self.qr, self.cr)

        # Apply single‑qubit rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Apply two‑qubit entanglement
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        circuit.measure(self.qr, self.cr)
        return circuit

    def _expectation_z(self, counts: dict) -> np.ndarray:
        """Compute the expectation value of Z for each qubit from measurement counts."""
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])

        expectations = []
        for qubit in range(self.n_qubits):
            # Extract bit value for this qubit
            bit_values = ((states >> qubit) & 1).astype(int)
            z_vals = 1 - 2 * bit_values  # map 0->+1, 1->-1
            expectations.append(np.sum(z_vals * probs))
        return np.array(expectations)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Execute the quantum circuit and compute attention‑weighted inputs.

        Parameters
        ----------
        rotation_params : np.ndarray
            Array of shape (3 * n_qubits,) specifying RX, RY, RZ angles.
        entangle_params : np.ndarray
            Array of shape (n_qubits - 1,) specifying CRX angles.
        inputs : np.ndarray
            Input matrix of shape (batch, embed_dim).

        Returns
        -------
        np.ndarray
            Attention‑weighted representation of the inputs.
        """
        batch_size = inputs.shape[0]
        out = np.zeros_like(inputs, dtype=np.float32)

        for i in range(batch_size):
            circuit = self._build_circuit(rotation_params, entangle_params)
            job = execute(circuit, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(circuit)

            # Convert measurement counts to attention weights
            attn_weights = self._expectation_z(counts)
            # Normalise to a probability distribution
            attn_weights = np.clip(attn_weights, 0, None)
            if attn_weights.sum() == 0:
                attn_weights = np.ones_like(attn_weights)
            attn_weights /= attn_weights.sum()

            # Weighted sum of the input features
            out[i] = attn_weights * inputs[i]

        return out


__all__ = ["SelfAttentionHybrid"]
