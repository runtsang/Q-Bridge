"""Quantum sampler with parameter‑shift gradients and multi‑observable support."""
from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import Probability

class ExtendedSamplerQNN:
    """
    A variational quantum sampler that can generate probability
    distributions and compute gradients via the parameter‑shift rule.
    """

    def __init__(self,
                 dev: qml.Device,
                 input_dim: int = 2,
                 weight_dim: int = 4,
                 observable: str | None = None) -> None:
        self.dev = dev
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.observable = observable or "PauliZ"

        # Parameter vectors
        self.inputs = qml.numpy.array([0.0] * input_dim)
        self.weights = qml.numpy.array([0.0] * weight_dim)

        # Build circuit
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            # Encode inputs
            for i in range(self.input_dim):
                qml.RY(inputs[i], wires=i)
            # Entangling layer
            qml.CNOT(0, 1)
            # Parameterised rotations
            for i in range(self.weight_dim):
                qml.RY(weights[i], wires=i % 2)
            # Entangling again
            qml.CNOT(0, 1)
            # Measurement
            return qml.probs(wires=[0, 1])  # 4‑dimensional probability vector
        return circuit

    def sample(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Return the probability distribution for given parameters."""
        self.inputs = inputs
        self.weights = weights
        return self.circuit(self.inputs, self.weights)

    def gradient(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute the gradient of the probability vector w.r.t. weights."""
        self.inputs = inputs
        self.weights = weights
        probs = self.circuit(self.inputs, self.weights)
        grad = qml.grad(self.circuit)(self.inputs, self.weights)
        return grad

__all__ = ["ExtendedSamplerQNN"]
