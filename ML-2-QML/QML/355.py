"""SamplerQNN for quantum probabilistic modelling.

This module defines SamplerQNN, a quantum neural network that mirrors the
classical SamplerQNN but uses a parameterised quantum circuit.  Enhancements
over the seed include:

* Three qubits and a three‑layer entangling Ansatz.
* Explicit state‑vector sampling via Aer or StatevectorSampler.
* Convenience methods for sampling and probability extraction.
* A KL‑divergence helper for evaluation against a target distribution.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import Aer
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN


class SamplerQNN(QiskitSamplerQNN):
    """Extended quantum sampler network based on Qiskit Machine Learning."""

    def __init__(
        self,
        qubits: int = 3,
        layers: int = 3,
        seed: int | None = None,
        backend: str = "statevector_simulator",
    ) -> None:
        """
        Construct a parameterised quantum circuit with the given depth.

        Args:
            qubits: Number of qubits in the Ansatz.
            layers: Number of RY‑CX entangling layers.
            seed: Random seed for circuit initialization.
            backend: Aer backend name for state‑vector sampling.
        """
        # Parameter vectors
        self.input_params = ParameterVector("input", qubits)
        self.weight_params = ParameterVector("weight", qubits * layers * 2)

        # Build the circuit
        qc = QuantumCircuit(qubits)
        # Input rotations
        for i in range(qubits):
            qc.ry(self.input_params[i], i)

        # Entangling layers
        for l in range(layers):
            idx = l * qubits * 2
            # First set of RY gates
            for i in range(qubits):
                qc.ry(self.weight_params[idx + i], i)
            # CX entanglement
            for i in range(qubits - 1):
                qc.cx(i, i + 1)
            qc.cx(qubits - 1, 0)
            # Second set of RY gates
            for i in range(qubits):
                qc.ry(self.weight_params[idx + qubits + i], i)

        # Instantiate the sampler primitive
        simulator = Aer.get_backend(backend)
        sampler = StatevectorSampler(simulator=simulator)

        super().__init__(
            circuit=qc,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=sampler,
        )

    def get_probabilities(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit and return the probability of measuring each basis state.

        Args:
            inputs: Array of shape (n_samples, qubits) with rotation angles in radians.

        Returns:
            Array of shape (n_samples, 2**qubits) with state‑vector probabilities.
        """
        if len(inputs.shape) == 1:
            inputs = inputs[None, :]
        result = super().sample(inputs, shots=0)
        return result["probabilities"]

    def sample(self, inputs: np.ndarray, num_shots: int = 1000) -> np.ndarray:
        """
        Draw samples from the quantum circuit for each input configuration.

        Args:
            inputs: Array of shape (n_samples, qubits) with rotation angles.
            num_shots: Number of shots per input.

        Returns:
            Array of shape (n_samples, num_shots) with integer basis state indices.
        """
        if len(inputs.shape) == 1:
            inputs = inputs[None, :]
        result = super().sample(inputs, shots=num_shots)
        return result["samples"]

    def kl_divergence(self, inputs: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Compute the KL divergence between the model output distribution and a target.

        Args:
            inputs: Array of shape (n_samples, qubits) with rotation angles.
            target: Array of shape (n_samples, 2**qubits) with target probabilities.

        Returns:
            Array of shape (n_samples,) with the KL divergence per sample.
        """
        probs = self.get_probabilities(inputs)
        eps = 1e-12
        return np.sum(target * (np.log(target + eps) - np.log(probs + eps)), axis=1)


__all__ = ["SamplerQNN"]
