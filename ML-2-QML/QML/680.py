"""Hybrid quantum convolutional filter with depth and parameter shift gradient estimation.

This module defines ConvEnhanced, a drop‑in replacement for the original QuanvCircuit.
It constructs a multi‑layer quantum circuit where each layer consists of a
parameterized rotation on each qubit followed by a random entangling block.
The depth parameter controls how many such layers are concatenated.  The
run method executes the circuit on a specified backend and returns the
average probability of measuring |1> across all qubits, matching the API
of the original QuanvCircuit.
"""

from __future__ import annotations

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
from typing import Iterable, Tuple, Callable

__all__ = ["ConvEnhanced"]


class ConvEnhanced:
    """
    Multi‑layer quantum convolutional filter with parameter shift gradient.
    """

    def __init__(
        self,
        kernel_size: int,
        depth: int = 1,
        backend=None,
        shots: int = 1000,
        threshold: float = 127,
    ) -> None:
        self.n_qubits = kernel_size ** 2
        self.depth = depth
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Build the circuit once; parameters are reused across layers
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        # Add layers
        for _ in range(depth):
            for i in range(self.n_qubits):
                self._circuit.rx(self.theta[i], i)
            self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        Execute the quantum circuit on classical data.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        # Compute average probability of |1> over all qubits
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

    def param_shift_gradient(self, data: np.ndarray) -> np.ndarray:
        """
        Estimate the gradient of the output with respect to each rotation
        parameter using the parameter‑shift rule.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        np.ndarray
            Gradient vector of shape (n_qubits,).
        """
        grad = np.zeros(self.n_qubits)
        shift = np.pi / 2
        base_output = self.run(data)

        for i in range(self.n_qubits):
            # Shift +pi/2
            self._circuit.assign_parameters({self.theta[i]: shift}, inplace=True)
            out_plus = self.run(data)
            # Shift -pi/2
            self._circuit.assign_parameters({self.theta[i]: -shift}, inplace=True)
            out_minus = self.run(data)
            # Restore original value
            self._circuit.assign_parameters({self.theta[i]: 0}, inplace=True)

            grad[i] = (out_plus - out_minus) / 2
        return grad
