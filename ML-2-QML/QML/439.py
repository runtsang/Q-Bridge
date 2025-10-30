"""Quantum convolutional filter with a learnable variational circuit.

The Conv() factory returns a ConvFilter instance that can be used as a drop‑in
replacement for the original quantum filter.  It uses a simple
parameter‑shared rotation‑only circuit followed by a ring of CNOTs to
introduce entanglement.  The ``run`` method accepts a 2‑D array,
maps its entries to rotation angles, executes the circuit on a simulator,
and returns the average probability of measuring ``|1>``.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from typing import Iterable
import math


class ConvFilter:
    """
    Quantum filter that implements a variational circuit with learnable
    rotation angles.  The circuit is parameter‑shared across all qubits
    to reduce the number of trainable parameters.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        depth: int = 2,
        shots: int = 1024,
        threshold: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the square kernel; determines the number of qubits.
        depth : int
            Number of variational layers.
        shots : int
            Number of shots for simulation.
        threshold : float
            Threshold used to map classical data to rotation angles.
        """
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.depth = depth
        self.shots = shots
        self.threshold = threshold

        # Parameter shared across all qubits
        self.params = [Parameter(f"θ_{i}") for i in range(self.n_qubits)]

        self.circuit = self._build_circuit()
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self) -> QuantumCircuit:
        """Construct a parameterised variational circuit."""
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)

        for d in range(self.depth):
            # Apply rotation around Y to all qubits
            for q in range(self.n_qubits):
                qc.ry(self.params[q], q)

            # Entangle neighbours with a simple ring of CNOTs
            for q in range(self.n_qubits):
                qc.cx(q, (q + 1) % self.n_qubits)

        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc

    def run(self, data: Iterable[float] | np.ndarray) -> float:
        """
        Execute the circuit on the provided data.

        Parameters
        ----------
        data : 2‑D array or list
            Input data; each element is mapped to a rotation angle
            (π if value > threshold else 0).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        # Flatten and reshape data to match qubit count
        arr = np.asarray(data, dtype=float).reshape(-1)
        if arr.size!= self.n_qubits:
            raise ValueError(f"Data must contain {self.n_qubits} elements.")

        # Bind parameters based on data
        theta_bind = {}
        for i, val in enumerate(arr):
            theta_bind[self.params[i]] = math.pi if val > self.threshold else 0.0

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[theta_bind],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Compute average probability of measuring |1> over all qubits
        total_ones = 0
        for bitstring, count in counts.items():
            total_ones += bitstring.count("1") * count

        return total_ones / (self.shots * self.n_qubits)

def Conv() -> ConvFilter:
    """Factory function that returns a ready‑to‑use ConvFilter instance."""
    return ConvFilter(kernel_size=2, depth=2, shots=1024, threshold=0.5)
