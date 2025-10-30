"""Quantum self‑attention with multi‑head parameterized circuits and configurable backend."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers import Backend
from qiskit.circuit.library import RX, RY, RZ, CRX


class SelfAttention:
    """
    Quantum multi‑head self‑attention circuit.

    Parameters
    ----------
    n_qubits : int
        Total number of qubits; must be divisible by num_heads.
    num_heads : int, default=1
        Number of attention heads.
    """

    def __init__(self, n_qubits: int, num_heads: int = 1):
        if n_qubits % num_heads!= 0:
            raise ValueError("n_qubits must be divisible by num_heads")
        self.n_qubits = n_qubits
        self.num_heads = num_heads
        self.head_qubits = n_qubits // num_heads

        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        """
        Construct a parameterized circuit for all heads.

        Parameters
        ----------
        rotation_params : np.ndarray
            Array of shape (n_qubits, 3) containing RX, RY, RZ angles per qubit.
        entangle_params : np.ndarray
            Array of shape (n_qubits - 1,) containing CRX angles for adjacent qubits.

        Returns
        -------
        QuantumCircuit
            Parameterized circuit ready for execution.
        """
        circuit = QuantumCircuit(self.qr, self.cr)

        # Rotation gates per qubit
        for q in range(self.n_qubits):
            rx, ry, rz = rotation_params[q]
            circuit.rx(rx, q)
            circuit.ry(ry, q)
            circuit.rz(rz, q)

        # Entanglement across heads (CRX between adjacent qubits)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend: Backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict[str, int]:
        """
        Execute the attention circuit on the specified backend.

        Parameters
        ----------
        backend : Backend
            Qiskit backend (e.g., Aer simulator or real device).
        rotation_params : np.ndarray
            Rotation angles; shape (n_qubits, 3).
        entangle_params : np.ndarray
            Entanglement angles; shape (n_qubits - 1,).
        shots : int, default=1024
            Number of measurement shots.

        Returns
        -------
        dict
            Measurement counts mapping bitstring to frequency.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


__all__ = ["SelfAttention"]
