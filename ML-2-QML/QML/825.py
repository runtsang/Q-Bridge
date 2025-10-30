import pennylane as qml
import pennylane.numpy as np

class SamplerQNNModel:
    """Quantum sampler network implemented with Pennylane.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must be >= 2).
    n_layers : int, optional
        Number of variational layers. Defaults to 2.
    """

    def __init__(self, n_qubits: int = 2, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            # Encode inputs as Ry rotations
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                # Entangling layer (chain of CNOTs)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Measurement in computational basis
            return qml.probs(wires=range(n_qubits))

        self.circuit = circuit

    def forward(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Return probability distribution over 2^n_qubits basis states."""
        return self.circuit(inputs, weights)

    def sample(self, inputs: np.ndarray, weights: np.ndarray,
               n_samples: int = 1) -> np.ndarray:
        """Draw samples from the quantum distribution."""
        probs = self.forward(inputs, weights)
        return np.random.choice(len(probs), size=n_samples, p=probs)

__all__ = ["SamplerQNNModel"]
