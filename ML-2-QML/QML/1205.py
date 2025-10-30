"""Quantum sampler network based on a variational circuit."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import Sampler as QSampler
from typing import Tuple


class SamplerQNN:
    """
    Variational quantum sampler with two qubits and two layers of entanglement.

    The circuit consists of:
      * Ry rotations on each qubit parameterised by the input.
      * Two entangling CX layers.
      * Two layers of Ry rotations parameterised by trainable weights.

    The class exposes a ``sample`` method that returns a probability
    distribution over the computational basis states and a helper to
    draw samples.
    """

    def __init__(self, num_layers: int = 2) -> None:
        self.num_layers = num_layers
        # Input and weight parameters
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4 * num_layers)
        # Build the variational circuit
        self.circuit = self._build_circuit()
        # Sampler primitive
        self.sampler = QSampler()
        # Wrap in Qiskit Machine Learning SamplerQNN for convenience
        self.qsampler_qnn = QSamplerQNN(
            circuit=self.circuit,
            input_params=self.inputs,
            weight_params=self.weights,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        # Input rotations
        qc.ry(self.inputs[0], 0)
        qc.ry(self.inputs[1], 1)
        # Entanglement layers
        for layer in range(self.num_layers):
            qc.cx(0, 1)
            qc.cx(1, 0)
            # Weight rotations
            w0 = self.weights[4 * layer]
            w1 = self.weights[4 * layer + 1]
            w2 = self.weights[4 * layer + 2]
            w3 = self.weights[4 * layer + 3]
            qc.ry(w0, 0)
            qc.ry(w1, 1)
            qc.cx(0, 1)
            qc.ry(w2, 0)
            qc.ry(w3, 1)
        return qc

    def probabilities(self, input_vals: Tuple[float, float]) -> np.ndarray:
        """
        Compute the probability distribution over |00>, |01>, |10>, |11>.

        Parameters
        ----------
        input_vals : tuple
            Two input angles for the Ry gates.

        Returns
        -------
        np.ndarray
            Array of shape (4,) with the probabilities.
        """
        bound_circuit = self.circuit.bind_parameters(
            {self.inputs[0]: input_vals[0], self.inputs[1]: input_vals[1]}
        )
        backend = Aer.get_backend("aer_simulator_statevector")
        job = execute(bound_circuit, backend)
        result = job.result()
        statevector = result.get_statevector()
        probs = np.abs(statevector) ** 2
        return probs

    def sample(self, input_vals: Tuple[float, float], num_shots: int = 1024) -> np.ndarray:
        """
        Draw samples from the quantum circuit.

        Parameters
        ----------
        input_vals : tuple
            Two input angles for the Ry gates.
        num_shots : int
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Array of shape (num_shots, 4) with one‑hot encoded samples.
        """
        probs = self.probabilities(input_vals)
        # Convert to one‑hot samples
        samples = np.random.choice(4, size=num_shots, p=probs)
        one_hot = np.eye(4)[samples]
        return one_hot

__all__ = ["SamplerQNN"]
