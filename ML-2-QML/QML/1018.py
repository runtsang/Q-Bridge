"""Quantum convolution circuit used by ConvEnhanced.

This module implements a reusable quantum circuit that maps a weighted patch
to a probability of measuring |1> across all qubits. The circuit is built
once during initialization and reused for all patches, which keeps the runtime
manageable. The design follows the original `Conv` interface but is
fully quantum, enabling experiments with different backends or circuit depths.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import execute
from qiskit.circuit.random import random_circuit

__all__ = ["QuanvCircuit"]

class QuanvCircuit:
    """
    Quantum convolution filter.

    Parameters
    ----------
    kernel_size : int
        Size of the sliding window (e.g., 3×3).
    backend : str or qiskit.providers.backend.Backend, optional
        Quantum backend. If None, a local Aer simulator is used.
    shots : int, default 200
        Number of shots for measurement.
    threshold : float, default 0.0
        Threshold used to map input values to gate parameters.
    depth : int, default 2
        Depth of the random circuit added after the parameterized RX gates.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        backend: str | qiskit.providers.backend.Backend | None = None,
        shots: int = 200,
        threshold: float = 0.0,
        depth: int = 2,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.depth = depth

        # Build quantum circuit
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [
            qiskit.circuit.Parameter(f"theta_{i}") for i in range(self.n_qubits)
        ]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        # Add random entangling layers
        self._circuit += random_circuit(self.n_qubits, depth, seed=42)
        self._circuit.measure_all()

        # Backend
        if backend is None:
            self.backend = qiskit.Aer.get_backend("qasm_simulator")
        else:
            self.backend = backend

    def run(self, vector: np.ndarray) -> float:
        """
        Execute the circuit for a single weighted patch.

        Parameters
        ----------
        vector : np.ndarray
            1‑D array of shape (n_qubits,) containing the weighted patch.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        if vector.shape!= (self.n_qubits,):
            raise ValueError(f"Expected vector of shape {(self.n_qubits,)}, got {vector.shape}")

        # Bind parameters based on threshold
        param_bind = {
            theta: np.pi if val > self.threshold else 0.0
            for theta, val in zip(self.theta, vector)
        }

        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        counts = result.get_counts(self._circuit)

        # Compute average probability of |1>
        total_ones = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * freq
        prob = total_ones / (self.shots * self.n_qubits)
        return prob
