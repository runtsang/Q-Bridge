"""Quantum implementation of the hybrid fully‑connected layer."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit.quantum_info import Statevector
from qiskit.providers.aer import AerSimulator


class FCLHybridQML:
    """
    Quantum‑backed counterpart to :class:`~ml.FCLHybrid`.
    Provides an ``run`` method that simulates a parameterised circuit
    and a ``kernel_matrix`` method based on Ry‑encoding overlap.
    """
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.backend = AerSimulator(name="aer_simulator")

    def _encode(self, x: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i, val in enumerate(x):
            qc.ry(val, i)
        return qc

    def kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return the absolute overlap between two encoded states."""
        sv_x = Statevector.from_instruction(self._encode(x))
        sv_y = Statevector.from_instruction(self._encode(y))
        return float(np.abs(np.vdot(sv_x.data, sv_y.data)))

    def kernel_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute Gram matrix between two datasets."""
        mat = np.zeros((len(a), len(b)))
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                mat[i, j] = self.kernel(x, y)
        return mat

    def _build_circuit(self, input_vals: np.ndarray, weight_params: np.ndarray) -> QuantumCircuit:
        """Build a parameterised circuit for the hybrid FCL."""
        qc = QuantumCircuit(self.n_qubits)
        # encode inputs
        for i, val in enumerate(input_vals):
            qc.ry(val, i)
        # encode weights (variational parameters)
        for i, w in enumerate(weight_params):
            qc.ry(w, i % self.n_qubits)
        # simple entangling layer
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        qc.measure_all()
        return qc

    def run(self, thetas: np.ndarray, input_vals: np.ndarray) -> np.ndarray:
        """
        Simulate the hybrid circuit with given parameters and inputs.
        Returns the expectation value of Z on the first qubit.
        """
        qc = self._build_circuit(input_vals, thetas)
        job = execute(qc, self.backend, shots=1024)
        result = job.result()
        counts = result.get_counts(qc)
        exp = 0.0
        for bitstring, cnt in counts.items():
            bit = int(bitstring[0])  # measurement of first qubit
            exp += (1 - 2 * bit) * cnt
        exp /= 1024
        return np.array([exp])


__all__ = ["FCLHybridQML"]
