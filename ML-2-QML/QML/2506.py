"""HybridSamplerQNN: Quantum sampler with classical post‑processing.

This module builds a parameterized quantum circuit that accepts two input
parameters and four weight parameters. The circuit is sampled using Qiskit's
StatevectorSampler, and the resulting probabilities are processed through
a classical fully connected layer to yield a scalar expectation value.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from typing import Iterable

class HybridSamplerQNN:
    """
    Quantum sampler that integrates a fully connected layer for post-processing
    of measurement outcomes. It is inspired by the SamplerQNN and FCL examples.
    """
    def __init__(self, shots: int = 1024) -> None:
        # Define input and weight parameters
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)

        # Build the quantum circuit
        self.circuit = QuantumCircuit(2)
        self.circuit.ry(self.inputs[0], 0)
        self.circuit.ry(self.inputs[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[0], 0)
        self.circuit.ry(self.weights[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[2], 0)
        self.circuit.ry(self.weights[3], 1)

        # Sampler primitive
        self.sampler = StatevectorSampler()
        # Wrap with Qiskit Machine Learning SamplerQNN
        self.sampler_qnn = QSamplerQNN(circuit=self.circuit,
                                       input_params=self.inputs,
                                       weight_params=self.weights,
                                       sampler=self.sampler)

        # Classical fully connected layer weights and bias
        self.fc_weights = np.random.randn(2, 1)
        self.fc_bias = np.random.randn(1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the quantum circuit with given parameters and process the
        measurement result through a fully connected layer.
        """
        # Bind parameters: first two are inputs, next four are weights
        param_bind = {
            self.inputs[0]: thetas[0],
            self.inputs[1]: thetas[1],
            self.weights[0]: thetas[2],
            self.weights[1]: thetas[3],
            self.weights[2]: thetas[4],
            self.weights[3]: thetas[5],
        }
        # Sample probabilities
        probs = self.sampler_qnn.run(param_bind)
        # Convert to numpy array
        probs_arr = np.array(probs)
        # Classical fully connected layer
        expectation = np.tanh(probs_arr @ self.fc_weights + self.fc_bias).mean()
        return np.array([expectation])
