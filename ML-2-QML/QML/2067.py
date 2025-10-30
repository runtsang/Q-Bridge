"""Quantum sampler network extending the original QNN helper.

- Parameterised variational circuit with configurable depth of entanglement.
- Uses Qiskit Machine Learning's SamplerQNN for state‑vector sampling
  and gradient‑based optimisation.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler


class SamplerQNN:
    """Quantum sampler neural network.

    Parameters
    ----------
    input_dim : int, default 2
        Number of input qubits / parameters.
    depth : int, default 1
        Number of entangling layers (CX gates).
    """

    def __init__(self, input_dim: int = 2, depth: int = 1) -> None:
        self.input_dim = input_dim
        self.depth = depth

        # Parameter vectors for inputs and trainable weights
        self.input_params = ParameterVector("input", input_dim)
        # Two Ry per qubit per layer + final Ry per qubit
        weight_count = input_dim * depth * 2 + input_dim
        self.weight_params = ParameterVector("weight", weight_count)

        # Build the variational circuit
        self.circuit = self._build_circuit()

        # Instantiate Qiskit SamplerQNN wrapper
        self.sampler_qnn = QiskitSamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=Sampler()
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Construct a variational circuit with alternating Ry and CX layers."""
        qc = QuantumCircuit(self.input_dim)

        # Input rotation layer
        for i, param in enumerate(self.input_params):
            qc.ry(param, i)

        # Entangling layers
        for l in range(self.depth):
            # Ry rotations for this layer
            layer_start = l * self.input_dim * 2
            for i in range(self.input_dim):
                qc.ry(self.weight_params[layer_start + i * 2], i)
            # CX gates in a ring topology
            for i in range(self.input_dim):
                qc.cx(i, (i + 1) % self.input_dim)

        # Final rotation layer
        final_start = self.depth * self.input_dim * 2
        for i in range(self.input_dim):
            qc.ry(self.weight_params[final_start + i], i)

        return qc

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Execute the sampler for a batch of inputs.

        Parameters
        ----------
        inputs : np.ndarray, shape (batch, input_dim)
            Input parameter values in radians.

        Returns
        -------
        np.ndarray
            Sample probabilities for each output basis state.
        """
        # Build a parameter dictionary
        param_dict = {p: v for p, v in zip(self.input_params, inputs.T)}
        # Execute sampler
        result = self.sampler_qnn.run(param_dict)
        return result["probs"]

__all__ = ["SamplerQNN"]
