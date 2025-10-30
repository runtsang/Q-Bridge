import pennylane as qml
import numpy as np

class SamplerQNNGen094:
    """
    Variational sampler implemented with Pennylane.
    Uses a two‑qubit circuit with parameterised rotations and CX entanglement.
    The circuit can return both probability distributions and discrete samples.
    Parameters:
        dev: Pennylane device (default: default.qubit)
        num_qubits: number of qubits (default: 2)
    """
    def __init__(self, dev: qml.Device = None, num_qubits: int = 2) -> None:
        self.dev = dev or qml.device("default.qubit", wires=num_qubits)
        self.num_qubits = num_qubits
        self.weights = np.random.uniform(0, 2 * np.pi, size=(num_qubits, 3))

    def circuit(self, inputs: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
        """Parameterized circuit returning statevector."""
        if weights is None:
            weights = self.weights
        for i in range(self.num_qubits):
            qml.RY(inputs[i], wires=i)
        for i in range(self.num_qubits):
            qml.RY(weights[i, 0], wires=i)
        qml.CNOT(wires=[0, 1])
        for i in range(self.num_qubits):
            qml.RZ(weights[i, 1], wires=i)
        qml.CNOT(wires=[1, 0])
        for i in range(self.num_qubits):
            qml.RX(weights[i, 2], wires=i)
        return qml.state()

    def probs(self, inputs: np.ndarray) -> np.ndarray:
        """Return probability distribution over 2^num_qubits outcomes."""
        state = self.circuit(inputs)
        probs = np.abs(state) ** 2
        return probs

    def sample(self, inputs: np.ndarray, num_samples: int = 1) -> np.ndarray:
        """Draw samples from the circuit's measurement distribution."""
        probs = self.probs(inputs)
        outcomes = np.random.choice(len(probs), size=num_samples, p=probs)
        return outcomes

    def set_weights(self, weights: np.ndarray) -> None:
        """Update the circuit's trainable weights."""
        self.weights = weights

__all__ = ["SamplerQNNGen094"]
