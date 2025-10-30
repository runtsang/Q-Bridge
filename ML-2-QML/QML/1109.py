import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class AdvancedSamplerQNN:
    """
    Quantum sampler using Pennylane’s strongly entangling layers.
    The circuit accepts 2 input parameters (rotation angles) and 4 weight parameters
    for the variational layers. It returns probabilities over the computational basis
    for 2 qubits, effectively a 2‑dimensional categorical distribution.
    """
    def __init__(self, device: qml.Device | None = None, seed: int | None = None) -> None:
        if device is None:
            device = qml.device("default.qubit", wires=2)
        self.dev = device
        if seed is not None:
            np.random.seed(seed)

        # Parameter placeholders
        self.input_params = pnp.array([0.0, 0.0])
        self.weight_params = pnp.array([0.0, 0.0, 0.0, 0.0])

        @qml.qnode(self.dev, interface="numpy")
        def circuit(inputs, weights):
            # Encode inputs as Ry rotations
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)
            # Entangling layer
            qml.CNOT(wires=[0, 1])
            # Variational rotations
            qml.RY(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(weights[2], wires=0)
            qml.RY(weights[3], wires=1)
            return qml.probs(wires=[0, 1])

        self._circuit = circuit

    def set_params(self, inputs: np.ndarray | list, weights: np.ndarray | list) -> None:
        """
        Update the circuit parameters.
        """
        self.input_params = np.array(inputs, dtype=float)
        self.weight_params = np.array(weights, dtype=float)

    def forward(self, inputs: np.ndarray | list) -> np.ndarray:
        """
        Return probability distribution over 2‑qubit computational basis.
        :param inputs: 1‑D array of length 2 (input angles)
        :return: 4‑element array of probabilities summing to 1.
        """
        probs = self._circuit(inputs, self.weight_params)
        # Collapse to 2‑dim by summing over basis states with same first qubit
        # e.g., |00> & |01> -> 0, |10> & |11> -> 1
        return np.array([probs[0] + probs[1], probs[2] + probs[3]])

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample from the quantum distribution using the current parameters.
        :param n_samples: Number of samples to draw
        :return: Array of shape (n_samples,) with integer labels 0 or 1.
        """
        probs = self.forward(self.input_params)
        return np.random.choice([0, 1], size=n_samples, p=probs)

    def get_qnode(self):
        """
        Expose the underlying QNode for hybrid training.
        """
        return self._circuit

__all__ = ["AdvancedSamplerQNN"]
