"""Quantum convolution module that can be used as a drop‑in replacement for Conv.

The ConvHybrid class implements a quantum filter that encodes a 2‑D kernel into a quantum circuit. The circuit applies RX rotations conditioned on pixel intensities, entangles the qubits with a random circuit, and measures all qubits. The returned feature is the average probability of measuring |1> across all qubits. The circuit is simulated with Qiskit Aer.

Typical usage::

    >>> from Conv__gen383 import ConvHybrid
    >>> conv = ConvHybrid(kernel_size=3, threshold=0.5, shots=1024)
    >>> val = conv.run(np.random.rand(3,3))

Author: GPT‑OSS‑20B
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit

__all__ = ["ConvHybrid"]


class ConvHybrid:
    """
    Quantum filter that encodes a 2‑D kernel into a quantum circuit.

    Parameters
    ----------
    kernel_size : int
        Size of the kernel (n_qubits = kernel_size**2).
    threshold : float
        Threshold for mapping pixel values to rotation angles.
    shots : int, default=1024
        Number of shots for the simulator.
    backend : qiskit.providers.backend.Backend, optional
        Quantum backend to use. If None, Aer QASM simulator is used.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        threshold: float = 0.5,
        shots: int = 1024,
        backend=None,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.n_qubits = kernel_size ** 2

        # Build the parameterized circuit
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        # Add a simple entangling layer
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, depth=2, entanglement="full")
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        Execute the quantum circuit on a 2‑D array of pixel values.

        Parameters
        ----------
        data : np.ndarray
            2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        flat = data.reshape(1, self.n_qubits)
        param_binds = []
        for row in flat:
            bind = {self.theta[i]: np.pi if val > self.threshold else 0.0 for i, val in enumerate(row)}
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self._circuit)

        # Compute average probability of |1> over all shots and qubits
        total_ones = 0
        for bitstring, cnt in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * cnt

        return total_ones / (self.shots * self.n_qubits)


def Conv() -> ConvHybrid:
    """Return a quantum convolution filter."""
    return ConvHybrid(kernel_size=2, threshold=0.5, shots=1024)
