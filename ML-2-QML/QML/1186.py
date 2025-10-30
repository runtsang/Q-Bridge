import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class SamplerQNNGen124:
    """
    Quantum sampler network built with Pennylane. Implements a
    parameterized circuit that accepts two input angles and four
    trainable weights. Supports evaluation of probability
    distribution via sampling and expectation value of PauliZ.
    """
    def __init__(self, num_qubits: int = 2, num_weights: int = 4, device_name: str = "default.qubit"):
        self.num_qubits = num_qubits
        self.num_weights = num_weights
        self.dev = qml.device(device_name, wires=num_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            # Encode inputs
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)
            # Entangling layer
            qml.CNOT(wires=[0, 1])
            # Parameterized rotation layers
            for i in range(num_weights):
                qml.RY(weights[i], wires=i % num_qubits)
            # Second entangling layer
            qml.CNOT(wires=[0, 1])
            # Output expectation values of PauliZ for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.circuit = circuit

    def probabilities(self, inputs: np.ndarray, weights: np.ndarray, num_samples: int = 1000) -> np.ndarray:
        """
        Return empirical probability distribution over computational basis
        states by sampling the circuit. The expectation values are treated
        as logits and exponentiated to form a softmax‑like distribution.
        """
        probs = np.zeros(2 ** self.num_qubits)
        for _ in range(num_samples):
            state = self.circuit(inputs, weights)
            logits = np.array(state)
            probs += np.exp(logits)
        probs /= num_samples
        return probs

    def expectation(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Return expectation values of PauliZ on each qubit."""
        return self.circuit(inputs, weights)

__all__ = ["SamplerQNNGen124"]
