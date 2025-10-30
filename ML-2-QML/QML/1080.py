"""Enhanced quantum sampler network using PennyLane variational circuit."""
import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class EnhancedSamplerQNN:
    """
    Variational quantum circuit that samples from a 2‑qubit probability distribution.
    The circuit consists of input rotations, alternating entangling layers and
    trainable rotation angles. The output probabilities are obtained by
    measuring in the computational basis.
    """
    def __init__(self, n_qubits: int = 2, n_layers: int = 2, device: str = "default.qubit") -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(device, wires=n_qubits)

        # Initialize trainable parameters
        self.weights = pnp.random.randn(n_layers, n_qubits, 3)  # rotations around X,Y,Z
        # Input parameters will be fed during evaluation
        self.input_params = pnp.array([0.0, 0.0])

    def circuit(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Quantum circuit with parameterized rotations and CNOT entanglement."""
        for i in range(self.n_layers):
            for q in range(self.n_qubits):
                qml.Rot(weights[i, q, 0], weights[i, q, 1], weights[i, q, 2], wires=q)
            # Entangle all qubits with a simple chain of CNOTs
            for q in range(self.n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        # Apply input rotations
        for q in range(self.n_qubits):
            qml.RY(inputs[q], wires=q)
        return qml.probs(wires=range(self.n_qubits))

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Return sampled probability distribution for given inputs."""
        probs = self.circuit(inputs, self.weights)
        return probs

    def sample(self, inputs: np.ndarray, n_shots: int = 1000) -> np.ndarray:
        """Draw samples from the quantum circuit using the device."""
        @qml.qnode(self.dev, interface="autograd")
        def qnode(inputs, weights):
            return self.circuit(inputs, weights)
        probs = qnode(inputs, self.weights)
        return np.random.choice(a=len(probs), size=n_shots, p=probs)

__all__ = ["EnhancedSamplerQNN"]
