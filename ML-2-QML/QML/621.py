import pennylane as qml
from pennylane import numpy as np

class SamplerQNNGen:
    """
    Variational sampler implemented with PennyLane.
    Uses two qubits and a trainable circuit that mirrors the classical
    architecture.  The ``sample`` method returns the probability
    distribution over the four computational basis states.
    """
    def __init__(self, dev_name: str = "default.qubit", shots: int = 1024):
        self.dev = qml.device(dev_name, wires=2, shots=shots)
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(weights[2], wires=0)
            qml.RY(weights[3], wires=1)
            return qml.probs(wires=[0, 1])
        self.circuit = circuit

    def sample(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Return probability distribution over 00, 01, 10, 11."""
        probs = self.circuit(inputs, weights)
        return probs

__all__ = ["SamplerQNNGen"]
