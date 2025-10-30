import pennylane as qml
import pennylane.numpy as np

class SamplerQNN:
    """Quantum sampler neural network with variational circuit."""
    def __init__(self, num_qubits: int = 2, num_layers: int = 2,
                 device: str = 'default.qubit'):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device(device, wires=num_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, inputs: np.ndarray, weights: np.ndarray) -> list[float]:
        # Input encoding
        for i in range(self.num_qubits):
            qml.RY(inputs[i], wires=i)
        # Variational layers
        for layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                qml.RY(weights[layer, qubit], wires=qubit)
            # Entangling pattern (ring)
            for qubit in range(self.num_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
            qml.CNOT(wires=[self.num_qubits - 1, 0])
        # Return full probability distribution
        return [qml.probs(wires=range(self.num_qubits))[i] for i in range(2 ** self.num_qubits)]

    def sample(self, inputs: np.ndarray, weights: np.ndarray,
               n_shots: int = 1000) -> list[str]:
        """Sample basis states according to the circuit’s probability distribution."""
        probs = self.qnode(inputs, weights)
        probs = np.array(probs)
        outcomes = np.random.choice(2 ** self.num_qubits, size=n_shots, p=probs)
        return [np.binary_repr(out, width=self.num_qubits) for out in outcomes]

    def get_probability_distribution(self, inputs: np.ndarray,
                                      weights: np.ndarray) -> np.ndarray:
        """Return the full probability distribution for given inputs and weights."""
        return np.array(self.qnode(inputs, weights))

__all__ = ["SamplerQNN"]
