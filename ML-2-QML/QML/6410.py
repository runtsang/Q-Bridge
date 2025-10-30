"""Quantum self‑attention block implemented with Qiskit.

The circuit applies a layer of arbitrary single‑qubit rotations followed by
controlled‑X entangling gates.  Expectation values of the Pauli‑Y operator on
each qubit are returned as a feature vector.  This block can be used as a
feature generator for classical learning algorithms.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator


class QuantumSelfAttention:
    """
    Quantum self‑attention module.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self._estimator = Estimator()

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        """Construct the variational circuit."""
        circuit = QuantumCircuit(self.qr, self.cr)
        # Rotation layer
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Entanglement layer
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)
            circuit.rz(entangle_params[i], i + 1)
        # No measurement needed – we use the Estimator to compute expectation values
        return circuit

    def _observables(self) -> list[SparsePauliOp]:
        """Pauli‑Y observable for each qubit."""
        obs = []
        for i in range(self.n_qubits):
            pauli_str = "I" * i + "Y" + "I" * (self.n_qubits - i - 1)
            obs.append(SparsePauliOp.from_list([(pauli_str, 1)]))
        return obs

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the circuit and return expectation values of Y on each qubit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Array of length ``3 * n_qubits`` containing rotation angles.
        entangle_params : np.ndarray
            Array of length ``n_qubits - 1`` containing entanglement angles.
        shots : int, optional
            Number of shots for the simulator (ignored by Estimator).

        Returns
        -------
        np.ndarray
            Expectation values of shape ``(n_qubits,)``.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        observables = self._observables()
        result = self._estimator.run(circuits=[circuit], observables=observables)
        # Extract expectation values
        expectations = [res.values[0].real for res in result]
        return np.array(expectations, dtype=np.float32)


def QuantumSelfAttentionFactory(n_qubits: int = 4) -> QuantumSelfAttention:
    """Convenience factory that returns a ready‑to‑use instance."""
    return QuantumSelfAttention(n_qubits=n_qubits)


__all__ = ["QuantumSelfAttention", "QuantumSelfAttentionFactory"]
