"""Quantum convolutional filter with a parameterised ansatz.

The ConvEnhanced class implements a quantum filter that can be trained via
classical optimisation. It replaces the random unitary from the seed with a
simple hardware‑efficient ansatz consisting of rotation layers and a CNOT
entanglement chain. The filter accepts a 2‑D image patch, maps pixel values
to rotation angles, and returns the average probability of measuring |1>
across all qubits.

The class exposes a `run` method that executes the circuit on a chosen
backend and a `parameters` property that allows external optimisation.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator
from typing import Iterable

class ConvEnhanced:
    """
    Quantum convolutional filter with a trainable ansatz.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the filter (number of qubits = kernel_size**2).
    backend : str | QuantumBackend, default None
        Backend to execute the circuit on. If None, an Aer simulator is used.
    shots : int, default 1000
        Number of shots for measurement.
    threshold : float, default 0.5
        Threshold for mapping pixel values to rotation angles.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend: str | None = None,
        shots: int = 1000,
        threshold: float = 0.5,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold

        # Backend
        if backend is None:
            self.backend = AerSimulator()
        else:
            self.backend = backend

        # Parameterised ansatz
        self.theta = [Parameter(f"θ{i}") for i in range(self.n_qubits)]
        self._circuit = QuantumCircuit(self.n_qubits, self.n_qubits)

        # Layer 1: RX rotations with trainable angles
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)

        # Entanglement chain (CNOTs)
        for i in range(self.n_qubits - 1):
            self._circuit.cx(i, i + 1)

        # Layer 2: RZ rotations with trainable angles
        for i in range(self.n_qubits):
            self._circuit.rz(self.theta[i], i)

        self._circuit.barrier()
        self._circuit.measure(range(self.n_qubits), range(self.n_qubits))

    @property
    def parameters(self) -> Iterable[Parameter]:
        """Return the list of trainable parameters."""
        return self.theta

    def run(self, data: np.ndarray) -> float:
        """
        Execute the filter on a 2‑D patch and return average |1> probability.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) with pixel values.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        flat = data.reshape(-1)
        param_bind = {}
        for idx, val in enumerate(flat):
            param_bind[self.theta[idx]] = np.pi if val > self.threshold else 0.0

        job = execute(
            self._circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        counts = result.get_counts(self._circuit)

        total_ones = 0
        total_shots = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * freq
            total_shots += freq
        avg_prob = total_ones / (total_shots * self.n_qubits)
        return avg_prob

__all__ = ["ConvEnhanced"]
