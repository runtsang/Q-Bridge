"""
Quantum‑variational counterpart to the ConvNetDual filter.
Replaces the single 2‑D convolution filter with a learnable
variational circuit that produces a scalar output.  The class
exposes a run() method that accepts a 2‑D array and returns
the average probability of measuring |1> across all qubits.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from qiskit.providers import BaseBackend
from typing import List

class ConvNetDual:
    """
    Small variational quantum filter that emulates the original
    Conv filter but uses a parameterised circuit.  The circuit
    has one qubit per pixel in the kernel and applies a layer of
    Ry rotations followed by a simple entanglement pattern.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 127,
        shots: int = 1024,
        backend: BaseBackend | None = None,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self._params = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Construct a simple variational circuit with Ry rotations
        followed by a ring of CNOTs."""
        qc = QuantumCircuit(self.n_qubits)
        # Parameterised Ry rotations
        for i, theta in enumerate(self._params):
            qc.ry(theta, i)

        # Simple entanglement: chain of CNOTs
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(self.n_qubits - 1, 0)  # close the ring

        qc.measure_all()
        return qc

    def _bind_parameters(self, data: np.ndarray) -> List[dict]:
        """Create a list of parameter bindings from the input data."""
        flat = data.flatten()
        binding = {}
        for i, val in enumerate(flat):
            binding[self._params[i]] = np.pi if val > self.threshold else 0.0
        return [binding]

    def run(self, data: np.ndarray | list) -> float:
        """
        Execute the variational circuit on a single kernel.

        Parameters
        ----------
        data : np.ndarray | list
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data_np = np.array(data, dtype=np.float32)
        param_binds = self._bind_parameters(data_np)

        job = execute(
            self._circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self._circuit)

        # Compute average probability of measuring |1>
        total_ones = 0
        total_counts = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * freq
            total_counts += freq

        avg_prob = total_ones / (self.shots * self.n_qubits)
        return avg_prob

    def run_batch(self, batch: List[np.ndarray | list]) -> List[float]:
        """
        Convenience wrapper that accepts a list of kernels.

        Parameters
        ----------
        batch : List[np.ndarray | list]
            List of 2‑D arrays each of shape (kernel_size, kernel_size).

        Returns
        -------
        List[float]
            List of scalar outputs for each kernel.
        """
        return [self.run(sample) for sample in batch]

def Conv() -> ConvNetDual:
    """
    Factory function that returns a ConvNetDual instance with
    default hyper‑parameters, mirroring the original Conv() API.
    """
    return ConvNetDual()

__all__ = ["ConvNetDual", "Conv"]
