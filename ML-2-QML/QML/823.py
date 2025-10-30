"""
EnhancedSamplerQNN – A parameterised quantum sampler with an expanded entangling schedule.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler


class EnhancedSamplerQNN:
    """
    Wraps a Qiskit SamplerQNN instance with a richer circuit:
    - Two input parameters (Ry rotations)
    - Two entangling layers (CX)
    - Four weight parameters (Ry rotations)
    Provides sampling and expectation utilities for hybrid workflows.
    """

    def __init__(
        self,
        input_dim: int = 2,
        weight_dim: int = 4,
        qubits: int = 2,
    ) -> None:
        self.input_params = ParameterVector("input", input_dim)
        self.weight_params = ParameterVector("weight", weight_dim)

        qc = QuantumCircuit(qubits)
        # Input encoding
        qc.ry(self.input_params[0], 0)
        qc.ry(self.input_params[1], 1)
        qc.cx(0, 1)

        # Variational block
        for i in range(weight_dim):
            qc.ry(self.weight_params[i], i % qubits)

        # Additional entangling layer
        qc.cx(0, 1)

        # Sampler primitive
        sampler = StatevectorSampler()

        # Construct the QNN
        self.sampler_qnn = SamplerQNN(
            circuit=qc,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=sampler,
        )

    def sample(
        self,
        inputs: np.ndarray,
        num_shots: int = 1024,
    ) -> dict[str, int]:
        """
        Return a dictionary of bitstring counts for the given input vector.
        """
        return self.sampler_qnn.sample(inputs=inputs, shots=num_shots)

    def expectation(
        self,
        inputs: np.ndarray,
        observable: str = "pauli_z",
    ) -> np.ndarray:
        """
        Compute the expectation value of a Pauli observable on the final state.
        """
        return self.sampler_qnn.expectation(inputs=inputs, observable=observable)

__all__ = ["EnhancedSamplerQNN"]
