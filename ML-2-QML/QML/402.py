"""Quantum self‑attention module using Qiskit.

This implementation uses a parameter‑shift compatible variational circuit
to generate a probability distribution that can be interpreted as
attention scores.  It supports a hybrid forward pass where the
quantum output is post‑processed with the classical input embeddings.

Usage
-----
>>> sa = SelfAttention(n_qubits=8, backend="qasm_simulator")
>>> out = sa(inputs, rot, ent, shots=2048)
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector

class SelfAttention:
    """Variational self‑attention circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must be even to allow pairwise entanglement).
    backend : str or qiskit.providers.Backend, default "qasm_simulator"
        Execution backend.
    """

    def __init__(self, n_qubits: int, backend: str | qiskit.providers.Backend = "qasm_simulator"):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend(backend) if isinstance(backend, str) else backend

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        """Construct a parameterised circuit for self‑attention."""
        circuit = QuantumCircuit(self.qr, self.cr)

        # Single‑qubit rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Controlled‑RX entanglement
        for i in range(0, self.n_qubits - 1, 2):
            circuit.crx(entangle_params[i // 2], i, i + 1)

        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the circuit and return a probability‑based attention map.

        Parameters
        ----------
        inputs : np.ndarray, shape (batch, seq_len, embed_dim)
            Classical embeddings used for post‑processing.
        rotation_params : np.ndarray
            Parameters for the rotation gates.
        entangle_params : np.ndarray
            Parameters for the entanglement gates.
        shots : int, default 1024
            Number of shots for the simulation.

        Returns
        -------
        np.ndarray, shape (batch, seq_len, embed_dim)
            Hybrid attention output.
        """
        batch, seq_len, embed_dim = inputs.shape
        assert embed_dim == self.n_qubits, "embed_dim must match n_qubits"

        # Build and execute circuit
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        counts = job.result().get_counts(circuit)

        # Convert counts to probabilities
        probs = np.zeros((self.n_qubits,))
        for bitstring, cnt in counts.items():
            idx = int(bitstring[::-1], 2)  # reverse due to qiskit convention
            probs[idx] = cnt / shots

        # Interpret probabilities as attention scores
        # Reshape to (batch, seq_len, embed_dim) via broadcasting
        scores = probs.reshape(1, 1, self.n_qubits)
        scores = np.broadcast_to(scores, (batch, seq_len, embed_dim))
        scores = scores / scores.sum(axis=-1, keepdims=True)  # softmax‑like

        # Hybrid output: weighted sum of inputs
        return np.sum(inputs * scores, axis=-1)

    @staticmethod
    def init_params(n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate random parameters for the variational circuit."""
        rng = np.random.default_rng()
        rot_shape = (3 * n_qubits,)
        ent_shape = (n_qubits // 2,)
        return rng.standard_normal(rot_shape), rng.standard_normal(ent_shape)

__all__ = ["SelfAttention"]
