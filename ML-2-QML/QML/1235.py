"""Quantum self‑attention with variational layers.

The class implements a variational circuit that mimics the
behaviour of a self‑attention block.  It can be executed on a
simulator or a real backend.  The interface mirrors the classical
version: ``run`` accepts a dictionary of parameters and returns a
measurement histogram.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute

class SelfAttentionPlus:
    """
    Variational quantum self‑attention.

    Parameters
    ----------
    n_qubits : int
        Number of qubits used for the circuit.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        """
        Build the variational circuit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Array of length ``3 * n_qubits`` containing RX, RY, RZ angles
            for each qubit.
        entangle_params : np.ndarray
            Array of length ``n_qubits - 1`` containing CP angles
            between adjacent qubits.
        """
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.cp(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, backend, params: dict, shots: int = 1024):
        """
        Execute the circuit on the given backend.

        Parameters
        ----------
        backend : qiskit.providers.Backend
            Backend to run the circuit on.
        params : dict
            Dictionary with keys ``rotation_params`` and ``entangle_params``.
        shots : int, default 1024
            Number of measurement shots.
        """
        rotation_params = params["rotation_params"]
        entangle_params = params["entangle_params"]
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        result = job.result()
        return result.get_counts(circuit)

__all__ = ["SelfAttentionPlus"]
