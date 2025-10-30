"""Quantum sampler network with a variational circuit and statevector sampling."""

import pennylane as qml
from pennylane import numpy as np

class SamplerQNNGen064:
    """Variational sampler that outputs a 2‑class probability distribution from a 2‑qubit circuit."""
    def __init__(self, dev=None):
        self.dev = dev or qml.device("default.qubit", wires=2)
        self.params = np.random.uniform(0, 2*np.pi, 4, requires_grad=True)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            # Encode inputs as rotations
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)
            # Entangle
            qml.CNOT(wires=[0, 1])
            # Parameterized layer
            qml.RY(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(weights[2], wires=0)
            qml.RY(weights[3], wires=1)
            # Measurement: probability of each computational basis state
            return qml.probs(wires=[0, 1])

        self.circuit = circuit

    def sample(self, inputs: np.ndarray) -> np.ndarray:
        """Return a 2‑class probability distribution for the given 2‑dimensional input."""
        probs = self.circuit(inputs, self.params)
        # Map the 4 outcome probabilities to two classes by grouping (00,01) vs (10,11)
        class_probs = np.array([probs[0] + probs[1], probs[2] + probs[3]])
        return class_probs / class_probs.sum()

    def loss(self, inputs: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Cross‑entropy loss between the sampler output and target distribution."""
        probs = self.sample(inputs)
        eps = 1e-12
        probs = np.clip(probs, eps, 1.0)
        target = np.clip(target, eps, 1.0)
        return -np.sum(target * np.log(probs))

def SamplerQNNGen064() -> SamplerQNNGen064:
    """Factory returning a fresh instance of the quantum sampler."""
    return SamplerQNNGen064()

__all__ = ["SamplerQNNGen064"]
