"""ConvEnhanced – a quantum convolution filter using a parameterized variational circuit."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator

class ConvEnhanced:
    """
    Quantum convolution filter that maps a 2D kernel to a probability value.
    Supports multi‑scale kernels and a trainable variational circuit.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 1024,
        backend: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        """
        Args:
            kernel_size: Size of the square kernel (number of qubits = kernel_size**2)
            shots: Number of measurement shots per evaluation
            backend: Name of Qiskit backend; if None, use Aer simulator
            threshold: Value to threshold classical data before encoding
        """
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold

        if backend is None:
            self.backend = AerSimulator()
        else:
            self.backend = Aer.get_backend(backend)

        # Parameterized variational circuit
        theta = [Parameter(f"θ_{i}") for i in range(self.n_qubits)]
        self.circuit = QuantumCircuit(self.n_qubits)
        for i, t in enumerate(theta):
            self.circuit.rx(t, i)
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        for i, t in enumerate(theta):
            self.circuit.rz(t, i)
        self.circuit.measure_all()

    def bind_parameters(self, data: np.ndarray) -> list[dict[Parameter, float]]:
        """
        Bind the data to the circuit parameters.
        Data is a flattened array of shape (n_qubits,).
        """
        binds = []
        for val in data:
            binds.append({t: np.pi if val > self.threshold else 0.0 for t in self.circuit.parameters})
        return binds

    def run(self, data: np.ndarray) -> float:
        """
        Evaluate the quantum filter on a single kernel patch.
        Args:
            data: 2D array of shape (kernel_size, kernel_size) with integer pixel values.
        Returns:
            float: average probability of measuring |1> across all qubits.
        """
        flat = data.flatten().astype(np.float32) / 255.0
        binds = self.bind_parameters(flat)

        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=binds)
        result = job.result()
        counts = result.get_counts(self.circuit)

        total_ones = 0
        for bitstring, count in counts.items():
            ones = bitstring.count('1')
            total_ones += ones * count
        prob = total_ones / (self.shots * self.n_qubits)
        return prob

__all__ = ["ConvEnhanced"]
