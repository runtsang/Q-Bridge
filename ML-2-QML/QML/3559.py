"""Hybrid quantum self‑attention with quanvolution encoding.

This module defines a quantum self‑attention block that first encodes a
2×2 input patch into qubits via a quanv circuit, then applies
parameter‑dependent rotations and controlled‑X entanglement.  The
resulting measurement statistics are interpreted as attention weights.
The construction follows the `SelfAttention.py` quantum seed, enriched
with the `Conv.py` quanv filter.  It uses Qiskit and the Aer simulator
for execution, making it runnable on local or cloud backends.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.random import random_circuit


class QuanvCircuit:
    """2×2 quanvolution filter circuit."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Run the quanv circuit on a 2×2 input patch."""
        data_flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data_flat:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        # Compute average number of |1> across all qubits
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)


class HybridQuantumSelfAttention:
    """Quantum self‑attention combining quanv encoding and parameterized gates."""
    def __init__(self, n_qubits: int = 4, kernel_size: int = 2, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.quanv = QuanvCircuit(kernel_size, self.backend, shots, threshold=127)

        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)

        # Quanv encoding
        circuit.append(self.quanv._circuit, self.qr)

        # Rotation layers
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Entanglement
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        """
        Execute the quantum self‑attention circuit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Flat array of length 3 * n_qubits.
        entangle_params : np.ndarray
            Flat array of length n_qubits - 1.
        shots : int
            Number of shots for simulation.

        Returns
        -------
        dict
            Measurement counts interpreted as attention weights.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)


__all__ = ["HybridQuantumSelfAttention"]
