"""ConvGen275: a quantum convolutional filter using a variational circuit.

The module implements a drop‑in replacement for the original Conv class, but the
forward pass is executed on a quantum simulator.  The circuit consists of a
parameterized rotation on each qubit followed by a small entangling layer.
The input data is encoded by mapping pixel values to rotation angles
(π if the pixel is above `threshold`, 0 otherwise).  The output is the
average probability of measuring |1⟩ across all qubits.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit import execute, Aer

class ConvGen275:
    """
    Quantum convolutional filter.

    Parameters
    ----------
    kernel_size : int
        Size of the filter (must be a perfect square).
    threshold : float
        Pixel threshold for encoding (default 0.5).
    shots : int
        Number of shots for the simulator.
    backend : qiskit.providers.Backend
        Quantum backend; defaults to Aer qasm simulator.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.5,
        shots: int = 1024,
        backend: qiskit.providers.Backend | None = None,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Create a parameterized circuit
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        for i, th in enumerate(self.theta):
            self.circuit.rx(th, i)
        # Add a small entangling layer
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        Execute the quantum circuit on the provided data.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape (kernel_size, kernel_size) with pixel values.

        Returns
        -------
        float
            Average probability of measuring |1⟩ across all qubits.
        """
        if data.shape!= (self.kernel_size, self.kernel_size):
            raise ValueError(
                f"Input data must have shape {(self.kernel_size, self.kernel_size)}"
            )
        # Flatten data
        flat = data.flatten()
        # Encode data into rotation angles
        bind_dict = {th: (np.pi if val > self.threshold else 0.0) for th, val in zip(self.theta, flat)}
        # Execute circuit
        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[bind_dict],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        # Compute average probability of |1⟩
        total = 0
        for bitstring, count in counts.items():
            ones = bitstring.count("1")
            total += ones * count
        avg_prob = total / (self.shots * self.n_qubits)
        return avg_prob

__all__ = ["ConvGen275"]
