"""HybridSamplerQNN: Pure quantum implementation of a sampler network.

This module exposes a lightweight class that constructs a Qiskit
``SamplerQNN`` with a simple two‑qubit parameterised circuit.  The class
provides a ``forward`` method that accepts a NumPy array of input
parameters and returns the sampled probability distribution.  It is
designed to be used as the quantum component in a hybrid architecture.
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN


class HybridSamplerQNN:
    """Pure quantum sampler network based on Qiskit’s SamplerQNN."""

    def __init__(self, input_dim: int = 2, weight_dim: int = 4, output_shape: int = 2):
        """
        Parameters
        ----------
        input_dim : int
            Number of input parameters that are mapped to the circuit qubits.
        weight_dim : int
            Number of trainable weight parameters in the circuit.
        output_shape : int
            Shape of the probability distribution returned by the sampler.
        """
        inputs = ParameterVector("x", input_dim)
        weights = ParameterVector("w", weight_dim)

        qc = QuantumCircuit(input_dim)

        # Encode inputs as Ry rotations on each qubit
        for i in range(input_dim):
            qc.ry(inputs[i], i)

        # Entangling layer
        qc.cx(0, 1)

        # Apply weight rotations (interleaved to keep the circuit depth low)
        for i in range(weight_dim):
            qc.ry(weights[i], i % input_dim)

        sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=sampler,
        )
        self.output_shape = output_shape

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the sampler.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (input_dim,) containing the input parameters.

        Returns
        -------
        probs : np.ndarray
            Probability distribution over ``output_shape`` outcomes.
        """
        return self.qnn.forward(inputs)

__all__ = ["HybridSamplerQNN"]
