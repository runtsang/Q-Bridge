"""Hybrid self‑attention: quantum branch.

This module defines a UnifiedSelfAttention class that implements a
variational quantum circuit performing a self‑attention‑style
operation.  The circuit is built from rotation and entanglement
parameters and is executed on a simulator or real backend.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def SelfAttention():
    """Return a quantum self‑attention module."""
    return UnifiedSelfAttention(n_qubits=4)

class UnifiedSelfAttention:
    """Quantum self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits used to encode the input and perform the
        attention‑style variational circuit.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        """
        Build a parameterised circuit from rotation and entanglement
        parameters.  The circuit follows the pattern used in the
        reference SelfAttention.py: Rx, Ry, Rz rotations followed by
        controlled‑RZ gates between neighbouring qubits.
        """
        circuit = QuantumCircuit(self.qr, self.cr)
        # Apply single‑qubit rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
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
    ) -> dict:
        """
        Execute the quantum circuit and return measurement counts.

        Parameters
        ----------
        backend : qiskit.providers.Backend
            Backend to run the circuit on.
        rotation_params : np.ndarray
            Array of shape (3 * n_qubits,) containing rotation angles.
        entangle_params : np.ndarray
            Array of shape (n_qubits - 1,) containing entanglement angles.
        shots : int, optional
            Number of shots for the execution.

        Returns
        -------
        dict
            Dictionary of measurement counts.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend=backend, shots=shots)
        return job.result().get_counts(circuit)

    def expectation_z(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Compute the expectation value of Pauli‑Z on each qubit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles.
        entangle_params : np.ndarray
            Entanglement angles.
        shots : int
            Number of shots.

        Returns
        -------
        np.ndarray
            Expectation values of shape (n_qubits,).
        """
        counts = self.run(self.backend, rotation_params, entangle_params, shots)
        probs = {int(k, 2): v / shots for k, v in counts.items()}
        exp_vals = np.zeros(self.n_qubits)
        for bitstring, p in probs.items():
            for q in range(self.n_qubits):
                bit = (bitstring >> (self.n_qubits - 1 - q)) & 1
                exp_vals[q] += p * (1 if bit == 0 else -1)
        return exp_vals

__all__ = ["UnifiedSelfAttention", "SelfAttention"]
