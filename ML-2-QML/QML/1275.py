"""Enhanced quantum convolutional filter with variational circuit.

This module defines Conv, a drop-in replacement for the original quantum filter.
It uses a parameterized circuit with entanglement and measurement to produce a scalar activation.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit

class Conv:
    """
    Quantum convolutional filter.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend: str | None = None,
        shots: int = 1024,
        threshold: float = 0.0,
        entanglement: str = "full",
        circuit_depth: int = 2,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = Aer.get_backend(backend or "qasm_simulator")
        self.entanglement = entanglement
        self.circuit_depth = circuit_depth
        self._build_circuit()

    def _build_circuit(self) -> None:
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        # Apply rotation gates
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        # Entanglement pattern
        if self.entanglement == "full":
            for i in range(self.n_qubits - 1):
                self.circuit.cx(i, i + 1)
        elif self.entanglement == "circular":
            for i in range(self.n_qubits):
                self.circuit.cx(i, (i + 1) % self.n_qubits)
        elif self.entanglement == "random":
            self.circuit += random_circuit(self.n_qubits, self.circuit_depth)
        else:
            # No entanglement
            pass
        # Measurement
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        Run the quantum circuit on classical data.

        Parameters
        ----------
        data : np.ndarray
            2D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {self.theta[i]: (np.pi if val > self.threshold else 0) for i, val in enumerate(dat)}
            param_binds.append(bind)
        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        total_ones = 0
        total_counts = 0
        for bitstring, count in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * count
            total_counts += count
        avg_prob = total_ones / (total_counts * self.n_qubits)
        return avg_prob

    def get_circuit(self) -> QuantumCircuit:
        """
        Return the underlying quantum circuit.
        """
        return self.circuit

__all__ = ["Conv"]
