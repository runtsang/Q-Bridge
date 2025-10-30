"""Quantum hybrid convolution + self‑attention circuit.

The circuit first applies data‑dependent RX rotations to each qubit
(analogous to a convolutional filter) and then executes a chain of
controlled‑RX gates that encode a self‑attention‑style entanglement.
The module exposes a ``run`` method that accepts a 2‑D NumPy array
and returns the average probability of measuring |1> across all qubits.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit.random import random_circuit


class QuantumConvAttention:
    """Quantum circuit combining convolution‑style rotations and
    attention‑style entanglement."""
    def __init__(
        self,
        kernel_size: int = 2,
        backend=None,
        shots: int = 1024,
        conv_threshold: float = 0.0,
        attention_n_qubits: int = 4,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.conv_threshold = conv_threshold
        # Rotation parameters for each data qubit
        self.rotation_params = np.random.uniform(0, 2 * np.pi, self.n_qubits)
        # Entanglement parameters for attention block
        self.entangle_params = np.random.uniform(0, 2 * np.pi, attention_n_qubits - 1)
        self._build_circuit()

    def _build_circuit(self) -> None:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        self.circuit = QuantumCircuit(qr, cr)
        # Convolution‑style rotations
        for i in range(self.n_qubits):
            self.circuit.rx(self.rotation_params[i], i)
        self.circuit.barrier()
        # Attention‑style entanglement
        for i in range(len(self.entangle_params)):
            self.circuit.crx(self.entangle_params[i], i, i + 1)
        self.circuit.measure(qr, cr)

    def run(self, data: np.ndarray) -> float:
        """Execute the hybrid circuit on the supplied data.

        Args:
            data: 2‑D array with shape (kernel_size, kernel_size).

        Returns:
            float: average probability of measuring |1> across qubits.
        """
        data_flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for sample in data_flat:
            bind = {}
            for idx, val in enumerate(sample):
                bind[self.circuit.parameters[idx]] = (
                    np.pi if val > self.conv_threshold else 0
                )
            param_binds.append(bind)
        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.circuit)
        # Compute average probability of |1>
        total_ones = 0
        for bitstring, count in result.items():
            total_ones += count * bitstring.count("1")
        return total_ones / (self.shots * self.n_qubits)

def ConvAttentionQ():
    """Factory returning a QuantumConvAttention instance."""
    return QuantumConvAttention()

__all__ = ["QuantumConvAttention", "ConvAttentionQ"]
