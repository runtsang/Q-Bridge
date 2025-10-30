import pennylane as qml
import numpy as np

class SamplerQNNGen:
    """
    A quantum sampler network implemented with Pennylane.
    The circuit consists of two qubits with parameterized rotations and a CNOT entangling layer.
    """

    def __init__(self, num_qubits: int = 2, num_layers: int = 2):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.weights = np.random.randn(num_layers, num_qubits, 3)  # 3 angles per qubit per layer

    def circuit(self, inputs: np.ndarray, weights: np.ndarray):
        for i in range(self.num_layers):
            for q in range(self.num_qubits):
                qml.RY(inputs[q], wires=q)
                qml.RZ(weights[i, q, 0], wires=q)
                qml.RX(weights[i, q, 1], wires=q)
                qml.RZ(weights[i, q, 2], wires=q)
            if i < self.num_layers - 1:
                qml.CNOT(wires=[0, 1])

    def get_probabilities(self, inputs: np.ndarray) -> np.ndarray:
        """Return the probability distribution over computational basis states."""
        @qml.qnode(self.dev)
        def circuit_node():
            self.circuit(inputs, self.weights)
            return qml.probs(wires=range(self.num_qubits))

        return circuit_node()

    def sample(self, inputs: np.ndarray, num_samples: int = 1) -> np.ndarray:
        """
        Sample measurement outcomes from the quantum circuit.
        """
        probs = self.get_probabilities(inputs)
        return np.random.choice(len(probs), size=num_samples, p=probs)
