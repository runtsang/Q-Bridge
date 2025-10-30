"""ConvEnhanced – a multi‑scale, hybrid convolutional filter for quantum workflows."""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import EfficientSU2
from qiskit import Aer, execute

class ConvEnhanced:
    """
    Quantum drop‑in replacement for the original Conv filter.
    Supports arbitrary kernel sizes and uses a parameter‑shaped variational circuit.
    The circuit encodes classical pixel values as rotation angles and outputs
    the average probability of measuring |1> across qubits.
    """

    def __init__(
        self,
        kernel_sizes: list[int] | tuple[int,...] = (2, 3),
        threshold: float = 0.0,
        shots: int = 1024,
        backend: str | qiskit.providers.BaseBackend = None,
    ) -> None:
        self.kernel_sizes = kernel_sizes
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.circuits = {}
        for ks in self.kernel_sizes:
            self.circuits[ks] = self._build_circuit(ks)

    def _build_circuit(self, kernel_size: int) -> QuantumCircuit:
        """
        Build a parameterized variational circuit for a given kernel size.
        """
        n_qubits = kernel_size ** 2
        qc = QuantumCircuit(n_qubits)
        # Parameters for data re-uploading
        params = [Parameter(f"theta_{i}") for i in range(n_qubits)]
        # Simple ansatz: RX rotations followed by a layer of CNOTs
        for i, p in enumerate(params):
            qc.rx(p, i)
        # Entangling layer
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        # Additional layer of rotations
        for i, p in enumerate(params):
            qc.ry(p, i)
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Run the quantum circuit on classical data.

        Args:
            data: 2D array with shape (kernel_size, kernel_size) for each kernel size.
        Returns:
            float: average probability of measuring |1> across qubits.
        """
        # data may contain multiple kernel sizes; we aggregate results
        results = []
        for ks in self.kernel_sizes:
            if data.shape!= (ks, ks):
                continue
            qc = self.circuits[ks]
            n_qubits = ks ** 2
            # Flatten data and encode as rotation angles
            flat = data.flatten()
            # Bind parameters: each qubit gets pi if pixel > threshold else 0
            param_bind = {f"theta_{i}": np.pi if val > self.threshold else 0 for i, val in enumerate(flat)}
            bound_qc = qc.bind_parameters(param_bind)
            job = execute(bound_qc, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(bound_qc)
            # Compute average probability of measuring |1>
            total_ones = 0
            for bitstring, count in counts.items():
                ones = bitstring.count("1")
                total_ones += ones * count
            avg_prob = total_ones / (self.shots * n_qubits)
            results.append(avg_prob)

        # Return average over all kernel sizes
        return float(np.mean(results)) if results else 0.0

__all__ = ["ConvEnhanced"]
