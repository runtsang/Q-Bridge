"""Quantum filter that emulates the behaviour of the original QuanvCircuit
while exposing a hybrid classical post‑processing step.

The circuit is constructed once per instance; the run method accepts a
2‑D array of shape (kernel_size, kernel_size) and returns the average
probability of measuring |1⟩ on all qubits, optionally passed through a
sigmoid threshold to match the classical counterpart.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit import Aer, execute

class HybridQuanvCircuit:
    """Drop‑in replacement for the original QuanvCircuit with a classical
    sigmoid post‑processing layer.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend: qiskit.providers.backend.BaseBackend | None = None,
        shots: int = 1024,
        threshold: float = 0.5,
    ) -> None:
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

        # Build a parameterised circuit that applies a random layer after
        # single‑qubit rotations encoding the input pixel values.
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        # Random two‑qubit entangling layer
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Execute the circuit on a single image patch.

        Args:
            data: 2‑D array of shape (kernel_size, kernel_size).

        Returns:
            float: average probability of measuring |1⟩ after a sigmoid
            post‑processing step with the configured threshold.
        """
        flat = np.reshape(data, (self.n_qubits,))
        # Bind parameters depending on pixel brightness relative to threshold
        param_bind = {}
        for i, val in enumerate(flat):
            param_bind[self.theta[i]] = np.pi if val > self.threshold else 0.0

        job = execute(
            self._circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result().get_counts(self._circuit)

        # Compute average |1⟩ probability
        total_ones = 0
        for bitstring, count in result.items():
            ones = sum(int(b) for b in bitstring)
            total_ones += ones * count
        prob = total_ones / (self.shots * self.n_qubits)

        # Classical sigmoid post‑processing to match the Conv filter
        return float(1 / (1 + np.exp(-prob)))

def Conv(kernel_size: int = 2, shots: int = 1024, threshold: float = 0.5):
    """Factory that returns a HybridQuanvCircuit instance."""
    return HybridQuanvCircuit(kernel_size=kernel_size, shots=shots, threshold=threshold)

__all__ = ["HybridQuanvCircuit", "Conv"]
