"""ConvAdvanced – quantum variational filter.

This module implements a class with the same public interface as the classical
ConvAdvanced but uses a parameterised quantum circuit as the filter.  The
circuit applies a rotation on each qubit conditioned on the input pixel
value, followed by a small entangling layer.  The expectation value of
measuring |1> on each qubit is averaged to give the filter output.

Author: gpt‑oss‑20b
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.circuit import Parameter
from qiskit.providers.aer import Aer
from typing import List

__all__ = ["ConvAdvanced"]


class ConvAdvanced:
    """Quantum variational filter with a 2‑D convolutional topology.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square filter; the number of qubits is ``kernel_size**2``.
    threshold : float, default 0.0
        Threshold for mapping classical pixel values to rotation angles.
    shots : int, default 100
        Number of shots used to estimate expectation values.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, shots: int = 100) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.n_qubits = kernel_size ** 2
        self.backend = Aer.get_backend("qasm_simulator")

        # Build a parameterised circuit with a single rotation per qubit
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"θ{i}") for i in range(self.n_qubits)]

        for i, theta in enumerate(self.theta):
            self.circuit.rx(theta, i)

        # Small entangling layer
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)

        # Measure all qubits
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Execute the quantum filter on a 2‑D array.

        Parameters
        ----------
        data : np.ndarray
            Array of shape ``(kernel_size, kernel_size)`` with pixel values in
            ``[0, 255]``.  Values above ``threshold`` are mapped to ``π``;
            otherwise to ``0`` for the rotation angle.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        flat = np.reshape(data, (1, self.n_qubits))

        # Bind parameters for each sample
        param_binds: List[dict] = []
        for sample in flat:
            bind = {theta: (np.pi if val > self.threshold else 0.0) for theta, val in zip(self.theta, sample)}
            param_binds.append(bind)

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.circuit)

        # Compute average number of |1> outcomes
        total_ones = 0
        total_counts = 0
        for bitstring, count in result.items():
            ones = sum(int(bit) for bit in bitstring)
            total_ones += ones * count
            total_counts += count

        avg_probability = total_ones / (total_counts * self.n_qubits)
        return avg_probability
