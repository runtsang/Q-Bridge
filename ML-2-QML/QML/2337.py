"""Quantum filter module used by ConvGen.

This module exposes a single class, QuantumConvFilter, that builds a
variational circuit similar to the quanvolution layer in the original
QML reference.  The circuit is parameterised by a threshold that
determines the rotation angles for each qubit.  The class provides a
``run`` method that accepts a 2‑D array and returns the mean probability
of measuring |1> across all qubits.

The implementation is intentionally lightweight and relies only on
Qiskit, making it easy to drop into any quantum‑aware pipeline.
"""

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit

class QuantumConvFilter:
    """Variational quantum filter for 2‑D data."""

    def __init__(
        self,
        kernel_size: int,
        shots: int = 100,
        threshold: float = 0.0,
        backend: str | None = None,
    ):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        if backend is None:
            self.backend = qiskit.Aer.get_backend("qasm_simulator")
        else:
            self.backend = qiskit.Aer.get_backend(backend)

        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray | list[list[float]]) -> float:
        """Run the circuit on a 2‑D array and return the mean |1> probability."""
        data_arr = np.array(data, dtype=np.float32).reshape(1, self.n_qubits)
        param_binds = []
        for sample in data_arr:
            bind = {}
            for i, val in enumerate(sample):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self._circuit)
        total_ones = 0
        total_counts = 0
        for bitstring, cnt in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * cnt
            total_counts += cnt
        return total_ones / (total_counts * self.n_qubits)

__all__ = ["QuantumConvFilter"]
