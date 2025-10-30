import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class SamplerQNNGen205:
    """
    Quantum sampler network using a two‑qubit variational circuit.
    Extends the original circuit by adding a second entangling layer
    and a measurement in the Z basis to produce a probability distribution.
    """

    def __init__(self,
                 input_dim: int = 2,
                 weight_dim: int = 4,
                 device_name: str = "default.qubit",
                 wires: int = 2,
                 shots: int = 1024) -> None:
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.shots = shots
        self.dev = qml.device(device_name, wires=wires, shots=shots)

        # Parameter vectors
        self.input_params = pnp.array([0.0] * input_dim)
        self.weight_params = pnp.array([0.0] * weight_dim)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            # Encode inputs with Ry rotations
            for i in range(input_dim):
                qml.Ry(inputs[i], wires=i)
            # First entangling layer
            qml.CNOT(wires=[0, 1])
            # Parameterized rotations
            for i in range(weight_dim):
                qml.Ry(weights[i], wires=i % wires)
            # Second entangling layer
            qml.CNOT(wires=[0, 1])
            # Measurement probabilities
            return qml.probs(wires=range(wires))

        self.circuit = circuit

    def __call__(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit and return the probability distribution.
        """
        probs = self.circuit(inputs, weights)
        return probs

    def sample(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Draw samples from the circuit output using the device's shot noise.
        """
        probs = self.__call__(inputs, weights)
        return np.random.choice(len(probs), size=self.shots, p=probs)

    def initialize(self, seed: int | None = None) -> None:
        """
        Randomly initialize the weight parameters.
        """
        rng = np.random.default_rng(seed)
        self.weight_params = rng.normal(size=self.weight_dim)

__all__ = ["SamplerQNNGen205"]
