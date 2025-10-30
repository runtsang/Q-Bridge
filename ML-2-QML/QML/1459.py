import pennylane as qml
import pennylane.numpy as np

class SamplerQNN:
    """Quantum sampler implemented with a PennyLane variational circuit."""
    def __init__(
        self,
        num_qubits: int = 2,
        device: str = "default.qubit",
        shots: int = 1000
    ) -> None:
        self.num_qubits = num_qubits
        self.device = qml.device(device, wires=num_qubits, shots=shots)
        self.weights_shape = (num_qubits, 3)  # one rotation per qubit
        self.circuit = qml.QNode(self._circuit, self.device)

    def _circuit(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Parameterized circuit producing a probability distribution."""
        # Encode inputs as rotation angles
        for i in range(self.num_qubits):
            qml.RY(inputs[i], wires=i)
        # Entangling layer
        for i in range(self.num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        # Variational rotations
        for i in range(self.num_qubits):
            qml.RX(weights[i, 0], wires=i)
            qml.RY(weights[i, 1], wires=i)
            qml.RZ(weights[i, 2], wires=i)
        # Measurement
        return qml.probs(wires=range(self.num_qubits))

    def forward(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Return the probability distribution given inputs and weights."""
        return self.circuit(inputs, weights)

    @staticmethod
    def random_weights(num_qubits: int) -> np.ndarray:
        """Generate random variational parameters."""
        return np.random.uniform(0, 2 * np.pi, size=(num_qubits, 3))

__all__ = ["SamplerQNN"]
