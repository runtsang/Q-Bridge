"""Variational self‑attention quantum module with depth and measurement‑based output.

The quantum implementation mirrors the classical API but replaces the
linear projections with parameter‑shifted rotations and introduces
a tunable entanglement depth. The output is a soft‑max over measurement
histograms to emulate a probability distribution, which is then used as
attention weights.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.fake_provider import FakeVigo

class QuantumSelfAttentionEnhanced:
    """Quantum self‑attention block based on a variational circuit."""

    def __init__(self, n_qubits: int, depth: int = 1, backend=None):
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits (must be >= 2).
        depth : int, default=1
            Number of variational layers.
        backend : qiskit.providers.backend, optional
            Backend to execute the circuit. If None, a fake simulator is used.
        """
        if n_qubits < 2:
            raise ValueError("n_qubits must be at least 2.")
        self.n_qubits = n_qubits
        self.depth = depth
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = backend or FakeVigo()

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        """Build a variational circuit with depth layers."""
        circuit = QuantumCircuit(self.qr, self.cr)

        # Parameter‑shifted rotations for each qubit
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Entanglement layers
        for d in range(self.depth):
            for i in range(self.n_qubits - 1):
                circuit.cx(i, i + 1)
            # Optional long‑range entanglement
            if self.n_qubits > 2:
                circuit.cx(self.n_qubits - 1, 0)

        # Entanglement parameters as additional rotations
        for i in range(self.n_qubits - 1):
            circuit.rx(entangle_params[i], i)
            circuit.rx(entangle_params[i + 1], i + 1)

        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the circuit and return a probability distribution over
        measurement outcomes, interpreted as attention weights.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles for the variational layers.
        entangle_params : np.ndarray
            Extra entanglement rotations.
        shots : int, default=1024
            Number of shots for the measurement histogram.

        Returns
        -------
        np.ndarray
            Soft‑max of the measurement histogram, shape (n_qubits,).
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=shots)
        counts = job.result().get_counts(circuit)

        # Convert counts to probability vector
        probs = np.zeros(self.n_qubits)
        for outcome, count in counts.items():
            idx = int(outcome[::-1], 2)
            probs[idx] = count / shots

        # Soft‑max to emulate attention weights
        softmax = np.exp(probs) / np.sum(np.exp(probs))
        return softmax

    def __repr__(self):
        return f"<QuantumSelfAttentionEnhanced n_qubits={self.n_qubits} depth={self.depth}>"

__all__ = ["QuantumSelfAttentionEnhanced"]
