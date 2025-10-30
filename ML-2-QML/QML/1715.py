import pennylane as qml
import numpy as np

class SamplerQNN:
    """
    Quantum sampler neural network using PennyLane.
    Implements a 2‑qubit variational circuit with input encoding and a trainable rotation layer.
    Provides `forward` to obtain measurement probabilities and `sample` to draw samples.
    """

    def __init__(self, device: qml.Device = None, seed: int = 42) -> None:
        self.dev = device or qml.device("default.qubit", wires=2, shots=1000)
        self.rng = np.random.default_rng(seed)

        # Initialize weights for the rotation layer
        self.weights = self.rng.uniform(0, 2 * np.pi, size=4)

        @qml.qnode(self.dev)
        def circuit(inputs: np.ndarray, weights: np.ndarray):
            # Input encoding
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)
            # Entangling layer
            qml.CNOT(wires=[0, 1])
            # Parameterized rotation layer
            qml.RY(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(weights[2], wires=0)
            qml.RY(weights[3], wires=1)
            return qml.probs(wires=[0, 1])  # 4 outcome probabilities

        self.circuit = circuit

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit and return the probability vector of the 4 basis states.
        """
        return self.circuit(inputs, self.weights)

    def sample(self, inputs: np.ndarray, num_samples: int = 1) -> np.ndarray:
        """
        Draw samples from the circuit's output distribution using the simulator's shots.
        Returns an array of shape (num_samples,) with integer outcome labels 0–3.
        """
        probs = self.forward(inputs)
        return self.rng.choice(4, size=num_samples, p=probs)
