"""Hybrid quantum‑convolution‑sampler circuit.

The module `ConvSamplerHybrid` builds a single Qiskit circuit that
combines:
* a convolution‑like parameterised block that acts on a kernel‑sized
  patch of data;
* a sampler block that produces a two‑dimensional probability
  distribution (mimicking a softmax).

The circuit is designed to be run on a Qiskit simulator or a real
backend, and can be used as a drop‑in replacement for the classical
`Conv` class in quantum‑aware training pipelines.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.random import random_circuit

__all__ = ["ConvSamplerHybrid"]


class ConvSamplerHybrid:
    """Quantum circuit that emulates a convolutional filter followed by a sampler."""

    def __init__(self, kernel_size: int = 2, shots: int = 500, threshold: float = 127):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold

        # Build convolution‑like block
        self.circuit = QuantumCircuit(self.n_qubits)
        self.input_params = ParameterVector("input", self.n_qubits)
        self.weight_params = ParameterVector("weight", 4)

        # Encode input as rotation angles
        for i in range(self.n_qubits):
            self.circuit.ry(self.input_params[i], i)

        # Simple entangling pattern
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)

        # Parameterised rotations (acts as learnable filter)
        self.circuit.ry(self.weight_params[0], 0)
        self.circuit.ry(self.weight_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[2], 0)
        self.circuit.ry(self.weight_params[3], 1)

        # Sampler block: measure all qubits
        self.circuit.measure_all()

        # Backend and sampler
        self.backend = Aer.get_backend("qasm_simulator")

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Args:
            data: 2‑D array of shape (kernel_size, kernel_size) with integer values.
        Returns:
            1‑D array of length 2: probabilities corresponding to two classes.
        """
        # Flatten input and bind parameters
        flat = data.reshape(1, self.n_qubits)
        param_binds = []
        for row in flat:
            bind = {}
            for i, val in enumerate(row):
                bind[self.input_params[i]] = np.pi if val > self.threshold else 0
            # Random weights for demonstration; in practice these would be trainable
            bind.update({
                self.weight_params[0]: np.pi / 4,
                self.weight_params[1]: np.pi / 4,
                self.weight_params[2]: np.pi / 4,
                self.weight_params[3]: np.pi / 4,
            })
            param_binds.append(bind)

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.circuit)

        # Convert counts to a probability vector over two classes
        # Class 0: majority of zeros, Class 1: majority of ones
        probs = np.zeros(2)
        for bitstring, freq in result.items():
            ones = sum(int(b) for b in bitstring)
            probs[ones % 2] += freq
        probs /= self.shots
        return probs
