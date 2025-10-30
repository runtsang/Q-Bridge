import pennylane as qml
import numpy as np

class SamplerQNN:
    """
    Quantum sampler network implemented with Pennylane.
    Provides a variational circuit that maps 2‑dimensional classical
    inputs to a probability distribution over 4 basis states.
    """
    def __init__(self) -> None:
        self.dev = qml.device("default.qubit", wires=2)
        # Initial random weights for the rotation layers
        self.weights = np.random.randn(4)
        # Define a QNode that accepts both inputs and trainable weights
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, inputs: np.ndarray, weights: np.ndarray) -> qml.measurements.Probs:
        """
        Parameterised circuit:
        - Input encoding with RY rotations
        - Two entangling layers (CNOTs)
        - Four trainable RY gates interleaved with entanglement
        """
        qml.RY(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)
        qml.CNOT(wires=[0, 1])

        # First layer of trainable rotations
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])

        # Second layer of trainable rotations
        qml.RY(weights[2], wires=0)
        qml.RY(weights[3], wires=1)

        return qml.probs(wires=[0, 1])

    def sample(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit for given classical inputs and return
        the probability distribution over the 4 basis states.
        """
        probs = self.qnode(inputs, self.weights)
        return probs

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Alias to sample() for functional usage.
        """
        return self.sample(inputs)

__all__ = ["SamplerQNN"]
