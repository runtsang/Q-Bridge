"""Quantum implementation of ConvFilter using a variational ansatz on Qiskit.

This module defines a drop‑in replacement for the original Conv class
that uses a parameterised quantum circuit to produce a scalar
activation probability.  The circuit is built with Qiskit and run on
the Aer simulator.  The rotation angles are trainable and can be
updated via a gradient‑based optimiser (e.g. via the parameter‑shift
rule, not shown here).
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import ParameterVector
from qiskit import Aer, execute
from typing import Tuple

__all__ = ["ConvFilter"]


class ConvFilter:
    """
    Variational quantum filter that emulates a convolutional kernel.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel (assumes square kernel).
    shots : int, default 1024
        Number of shots for each simulation.
    threshold : float, default 0.5
        Threshold used to decide whether to apply a π rotation.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 1024,
        threshold: float = 0.5,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size * kernel_size
        self.shots = shots
        self.threshold = threshold

        # Parameter vector for each qubit
        self.theta = ParameterVector("theta", self.n_qubits)

        # Build the variational circuit
        self.circuit = qiskit.QuantumCircuit(self.n_qubits, self.n_qubits)
        # Parameterised Ry rotations
        for i in range(self.n_qubits):
            self.circuit.ry(self.theta[i], i)
        # Simple entangling layer
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        # Measure all qubits
        self.circuit.measure(range(self.n_qubits), range(self.n_qubits))

        # Backend
        self.backend = Aer.get_backend("qasm_simulator")

    def run(self, data: np.ndarray | list | tuple) -> float:
        """
        Execute the circuit on the provided data.

        Parameters
        ----------
        data : array‑like of shape (kernel_size, kernel_size)
            Input image patch.

        Returns
        -------
        float
            Probability of measuring |1> averaged over all qubits.
        """
        # Flatten input and bind parameters
        flat = np.array(data, dtype=np.float32).flatten()
        bind_dict = {
            self.theta[i]: np.pi if flat[i] > self.threshold else 0.0
            for i in range(self.n_qubits)
        }

        job = execute(
            self.circuit.bind_parameters(bind_dict),
            self.backend,
            shots=self.shots,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Compute average probability of measuring |1>
        total_ones = 0
        for outcome, count in counts.items():
            total_ones += outcome.count("1") * count

        return total_ones / (self.shots * self.n_qubits)
