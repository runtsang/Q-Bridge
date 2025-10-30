"""
Quantum self‑attention that estimates the attention weights via a
parameterised quantum kernel.  The kernel is evaluated using a swap‑test
style circuit that compares the encoded query and key states.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import AerSimulator


class SelfAttentionQuantumHybrid:
    """
    Quantum self‑attention module that mirrors the classical interface.
    The attention weights are derived from a kernel estimated by a
    parameterised quantum circuit.  The circuit uses a simple
    rotation‑plus‑entangle ansatz and a swap‑test to compute the overlap
    between encoded query and key states.
    """

    def __init__(self, n_qubits: int, gamma: float = 1.0):
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits used to encode each embedding.
        gamma : float, default=1.0
            RBF width used to convert the quantum kernel into attention
            scores.
        """
        self.n_qubits = n_qubits
        self.gamma = gamma
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = AerSimulator()

    def _encode_data(self, circuit: QuantumCircuit, data: np.ndarray, offset: float = 0.0):
        """
        Encode a single embedding into the circuit via rotations.

        Parameters
        ----------
        circuit : QuantumCircuit
            The circuit to add gates to.
        data : ndarray of shape (n_qubits,)
            The embedding to encode.
        offset : float
            Optional offset to apply to all angles (used for the swap‑test).
        """
        for i in range(self.n_qubits):
            circuit.rx((data[i] + offset) * np.pi, i)
            circuit.ry((data[i] + offset) * np.pi, i)
            circuit.rz((data[i] + offset) * np.pi, i)

    def _swap_test_kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Estimate the kernel value k(x, y) = |⟨x|y⟩|² via a swap‑test circuit.

        Parameters
        ----------
        x, y : ndarray of shape (n_qubits,)

        Returns
        -------
        float
            Estimated kernel value in [0, 1].
        """
        circuit = QuantumCircuit(self.qr, self.cr)

        # Ancilla qubit for the swap test
        ancilla = QuantumRegister(1, "a")
        circuit.add_register(ancilla)

        # Prepare ancilla in |+⟩
        circuit.h(ancilla[0])

        # Encode data into the two registers
        self._encode_data(circuit, x, offset=0.0)
        self._encode_data(circuit, y, offset=0.0)

        # Controlled‑SWAP between the two data registers
        for i in range(self.n_qubits):
            circuit.cswap(ancilla[0], self.qr[i], self.qr[i])

        # Measure ancilla
        circuit.h(ancilla[0])
        circuit.measure(ancilla[0], self.cr[0])

        # Execute
        job = qiskit.execute(circuit, self.backend, shots=1024)
        result = job.result()
        counts = result.get_counts(circuit)
        prob_zero = counts.get("0", 0) / 1024.0
        # Swap‑test probability of ancilla in |0⟩ is (1 + |⟨x|y⟩|²)/2
        return 2 * prob_zero - 1

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute a quantum self‑attention step.

        Parameters
        ----------
        rotation_params : ndarray of shape (3 * n_qubits,)
            Rotation angles for the ansatz (mirrored from the classical API).
        entangle_params : ndarray of shape (n_qubits - 1,)
            Entangling angles (also only for API compatibility).
        inputs : ndarray of shape (N, n_qubits)
            Input embeddings.
        shots : int, default=1024
            Number of shots for the swap‑test estimation.

        Returns
        -------
        ndarray of shape (N, n_qubits)
            The attended embeddings.
        """
        n = inputs.shape[0]
        # Build kernel matrix using swap‑test
        kernel = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                kernel[i, j] = self._swap_test_kernel(inputs[i], inputs[j])

        # Convert kernel to attention scores
        scores = np.exp(-self.gamma * (1 - kernel))
        scores = scores / scores.sum(axis=-1, keepdims=True)

        # Weighted sum over the original embeddings
        return scores @ inputs


__all__ = ["SelfAttentionQuantumHybrid"]
