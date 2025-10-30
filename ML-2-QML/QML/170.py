"""Quantum filter implementation using Qiskit.

The filter implements a variational circuit with a layer of RX rotations followed by
an entangling layer and a measurement of the Z expectation value.  The parameters
are optimized using the parameter‑shift rule during training.  The circuit accepts
a 2‑D array of shape (kernel_size, kernel_size) and returns the average probability
of measuring |1> across all qubits.
"""

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector

__all__ = ["QuantumFilter"]


class QuantumFilter:
    """
    Variational quantum filter with trainable parameters.
    """

    def __init__(
        self,
        num_qubits: int,
        backend: str = "qasm_simulator",
        shots: int = 1024,
        threshold: float = 0.0,
    ):
        self.num_qubits = num_qubits
        self.threshold = threshold
        self.backend = Aer.get_backend(backend)
        self.shots = shots
        # Parameter vector
        self.params = ParameterVector("theta", length=num_qubits)
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        # RX rotations
        for i, p in enumerate(self.params):
            qc.rx(p, i)
        # Simple entanglement (nearest neighbour)
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)
        qc.barrier()
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on the given data.

        Args:
            data: 2‑D array of shape (k, k) with values in [0, 255].
        Returns:
            float: average probability of measuring |1> across all qubits.
        """
        # Flatten and map data to angles
        arr = data.flatten()
        param_binds = {self.params[i]: (np.pi if val > self.threshold else 0.0) for i, val in enumerate(arr)}
        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_binds],
        )
        result = job.result()
        counts = result.get_counts(self._circuit)
        # Compute expectation of Z (probability of |1>)
        total = 0.0
        for bitstring, cnt in counts.items():
            ones = bitstring.count("1")
            total += ones * cnt
        prob = total / (self.shots * self.num_qubits)
        return prob
