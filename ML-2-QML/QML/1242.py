"""
ConvEnhanced: variational quanvolution filter.

Implements a parameterized quantum circuit that replaces the random circuit from the seed. The circuit applies RX rotations parameterized by the input pixel values, an entangling layer, and measures all qubits. The output is the average probability of measuring |1> across the qubits.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.random import random_circuit
from qiskit.providers.aer import AerSimulator


class ConvEnhanced:
    """
    Variational quanvolution circuit.

    Parameters
    ----------
    kernel_size : int, default=2
        Size of the convolution kernel (determines number of qubits).
    shots : int, default=1024
        Number of shots for the simulation.
    threshold : float, default=0.5
        Threshold used to map pixel values to rotation angles.
    backend : qiskit.providers.backend.Backend, optional
        Backend to execute the circuit. Defaults to AerSimulator(statevector).
    """

    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 1024,
        threshold: float = 0.5,
        backend=None,
    ):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.backend = backend or AerSimulator(method="statevector")
        self.theta = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(self.theta[i], i)
        qc += random_circuit(self.n_qubits, 2)
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray | list[list[float]]) -> float:
        """
        Execute the circuit on 2‑D data.

        Parameters
        ----------
        data : array-like of shape (kernel_size, kernel_size)
            Input pixel values.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data_flat = np.reshape(data, (self.n_qubits,))
        param_binds = {}
        for i, val in enumerate(data_flat):
            angle = np.pi if val > self.threshold else 0.0
            param_binds[self.theta[i]] = angle

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_binds],
        )
        result = job.result()
        counts = result.get_counts(self._circuit)
        total_ones = sum(bitstring.count("1") * freq for bitstring, freq in counts.items())
        return total_ones / (self.shots * self.n_qubits)


def Conv() -> ConvEnhanced:
    """Return a ConvEnhanced instance."""
    return ConvEnhanced(kernel_size=2, shots=1024, threshold=0.5)
