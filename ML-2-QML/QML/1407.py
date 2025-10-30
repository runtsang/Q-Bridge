import pennylane as qml
import numpy as np

class SamplerQNN:
    """
    Quantum sampler network using Pennylane.

    Features:
    - 2‑qubit variational circuit with alternating RY and CNOT layers.
    - Separate input and trainable weight parameters.
    - State‑vector measurement of all basis probabilities.
    - Gradient‑friendly via Pennylane's qnode decorator.
    """
    def __init__(self, input_shape: int = 2, weight_dim: int = 4, seed: int = 42):
        self.input_shape = input_shape
        self.weight_dim = weight_dim
        self.dev = qml.device("default.qubit", wires=2, shots=None)
        np.random.seed(seed)
        self.weights = np.random.randn(self.weight_dim)

    def _circuit(self, inputs: np.ndarray, weights: np.ndarray):
        qml.RY(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(weights[2], wires=0)
        qml.RY(weights[3], wires=1)

    def probs(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        @qml.qnode(self.dev, interface="autograd")
        def qnode():
            self._circuit(inputs, weights)
            return qml.probs(wires=[0, 1])
        return qnode()

    def forward(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return self.probs(inputs, weights)

__all__ = ["SamplerQNN"]
