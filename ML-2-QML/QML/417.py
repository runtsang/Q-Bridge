import pennylane as qml
from pennylane import numpy as np
from pennylane.qinfo import entropy

class SamplerQNN:
    """
    Variational sampler using Pennylane.
    Implements a 3‑qubit circuit with entangling layers.
    Returns the probability distribution over the computational basis.
    """
    def __init__(self, n_qubits: int = 3, layers: int = 2):
        self.n_qubits = n_qubits
        self.layers = layers
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=1024)

        # Parameter shapes
        self.params = np.random.randn(layers, n_qubits)

        @qml.qnode(self.dev)
        def circuit(inputs, weights):
            # Encode inputs as Ry rotations
            for i, w in enumerate(inputs):
                qml.RY(w, wires=i)

            # Variational layers
            for l in range(layers):
                for q in range(n_qubits):
                    qml.RZ(weights[l, q], wires=q)
                # Entangling layer
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])

            return qml.probs(wires=range(n_qubits))

        self.circuit = circuit

    def sample(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit with given input angles and return the probability vector.
        """
        probs = self.circuit(inputs, self.params)
        return probs

    def entropy(self, inputs: np.ndarray) -> float:
        """
        Compute the von Neumann entropy of the output distribution.
        """
        probs = self.sample(inputs)
        return entropy(probs)

__all__ = ["SamplerQNN"]
