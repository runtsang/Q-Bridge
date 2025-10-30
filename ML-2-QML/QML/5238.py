"""Quantum‑classical hybrid self‑attention helper.

This file provides a Qiskit‑centric view of the hybrid module.  The
class `QuantumSelfAttentionHybrid` is defined to be importable in a
pure‑quantum context, exposing a `run` method that executes a
parameterised variational circuit and returns a probability
distribution over attention weights.  The quantum part mirrors the
classical self‑attention logic but operates on qubits, using
rotation and entanglement parameters that are clipped to keep the
circuit stable during training.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, assemble, transpile
from qiskit.providers.aer import AerSimulator
from typing import Tuple

class QuantumSelfAttentionHybrid:
    """Hybrid attention head that can be run on a Qiskit simulator."""
    def __init__(self, n_qubits: int = 4, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self._build_circuit()

    def _build_circuit(self):
        self.qr = QuantumRegister(self.n_qubits, "q")
        self.cr = ClassicalRegister(self.n_qubits, "c")
        self.circuit = QuantumCircuit(self.qr, self.cr)

        # Rotation parameters (3 per qubit)
        self.rotation_params = [self.circuit.rx(0, i) for i in range(self.n_qubits)]
        # Entanglement parameters (1 per adjacent pair)
        self.entangle_params = [self.circuit.cx(i, i+1) for i in range(self.n_qubits-1)]

        self.circuit.measure(self.qr, self.cr)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = None) -> np.ndarray:
        """
        Execute the quantum self‑attention circuit and return a probability
        distribution over qubits.

        Parameters
        ----------
        rotation_params : np.ndarray
            Flattened rotation angles (3 * n_qubits).
        entangle_params : np.ndarray
            Entanglement angles (n_qubits - 1).
        shots : int, optional
            Number of shots; defaults to the instance value.

        Returns
        -------
        np.ndarray
            Probability distribution of shape (n_qubits,).
        """
        shots = shots or self.shots

        # Clip parameters to avoid extreme values
        rotation_params = np.clip(rotation_params, -np.pi, np.pi)
        entangle_params = np.clip(entangle_params, -np.pi, np.pi)

        # Update circuit parameters
        for i, val in enumerate(rotation_params):
            self.circuit.data[i].operation.params = [val]
        for i, val in enumerate(entangle_params):
            self.circuit.data[self.n_qubits + i].operation.params = [val]

        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=shots)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts(self.circuit)

        probs = np.zeros(self.n_qubits)
        for state, cnt in counts.items():
            idx = int(state, 2)
            probs[idx] = cnt / shots
        return probs

__all__ = ["QuantumSelfAttentionHybrid"]
