"""Quantum variational sampler with two qubits and multiple entangling layers."""
from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class SamplerQNN:
    """A variational sampler that returns probabilities over 2‑bit strings."""

    def __init__(self, num_qubits: int = 2, num_layers: int = 3, device: qml.Device | None = None) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.device = device or qml.device("default.qubit", wires=num_qubits)
        # initialise parameters
        self.weights = pnp.random.randn(num_layers, num_qubits, 2, requires_grad=True)
        self.inputs = pnp.zeros((num_qubits,))

        # define the circuit
        @qml.qnode(self.device, interface="autograd")
        def circuit(inputs, weights):
            # input encoding
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)
            # variational layers
            for l in range(num_layers):
                for i in range(num_qubits):
                    qml.RY(weights[l, i, 0], wires=i)
                    qml.RZ(weights[l, i, 1], wires=i)
                # entanglement pattern (ring)
                for i in range(num_qubits):
                    qml.CNOT(wires=[i, (i + 1) % num_qubits])
            # measurement probabilities
            return qml.probs(wires=range(num_qubits))

        self.circuit = circuit

    def probs(self, inputs: np.ndarray) -> np.ndarray:
        """Return the probability distribution for the given inputs."""
        return self.circuit(inputs, self.weights)

    def sample(self, inputs: np.ndarray, num_shots: int = 1024) -> np.ndarray:
        """Draw samples from the circuit."""
        probs = self.probs(inputs)
        return np.random.choice(4, size=num_shots, p=probs)

    def loss(self, inputs: np.ndarray, target: np.ndarray) -> float:
        """Negative log‑likelihood loss."""
        probs = self.probs(inputs)
        return -np.sum(target * np.log(probs + 1e-9))

    def train(self, data: np.ndarray, targets: np.ndarray, lr: float = 0.01, epochs: int = 100):
        """Simple gradient‑descent training loop."""
        opt = qml.gradients.GradientDescentOptimizer(lr)
        for _ in range(epochs):
            for x, y in zip(data, targets):
                params = opt.step(lambda w: self.loss(x, y), self.weights)
                self.weights = params


__all__ = ["SamplerQNN"]
