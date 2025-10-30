import pennylane as qml
import numpy as np

class SamplerQNNEnhanced:
    """
    Variational sampler built with PennyLane that mirrors the classical
    SamplerQNNEnhanced.  Two qubits are used; the circuit contains
    alternating rotation and entangling layers.  The parameters are
    split into input (state preparation) and weight (variational) parts.
    The class provides a sample method that returns one‑hot encoded samples.
    """
    def __init__(self, n_qubits: int = 2, n_layers: int = 2, seed: int = 42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.seed = seed
        np.random.seed(seed)
        self.device = qml.device("default.qubit", wires=n_qubits, shots=1024)

    def circuit(self, inputs: np.ndarray, weights: np.ndarray):
        """Parameterized quantum circuit."""
        # Input encoding
        for i, w in enumerate(inputs):
            qml.RY(w, wires=i)
        # Variational layers
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                idx = layer * self.n_qubits + i
                qml.RY(weights[idx], wires=i)
            qml.CNOT(wires=[0, 1])
        # Measure first qubit only
        return qml.probs(wires=0)

    def build_qnode(self):
        @qml.qnode(self.device, interface="autograd")
        def qnode(inputs, weights):
            return self.circuit(inputs, weights)
        return qnode

    def sample(self, inputs: np.ndarray, weights: np.ndarray, n_samples: int = 1):
        """
        Draw samples from the quantum circuit for given inputs and weights.

        Args:
            inputs: array of shape (n_qubits,)
            weights: array of shape (n_qubits * n_layers,)
            n_samples: number of samples to draw

        Returns:
            array of shape (n_samples, n_qubits) with 0/1 outcomes
        """
        qnode = self.build_qnode()
        probs = qnode(inputs, weights)
        probs = np.array(probs)
        samples = np.zeros((n_samples, self.n_qubits), dtype=np.int32)
        for j in range(n_samples):
            choice = np.random.choice(len(probs), p=probs)
            outcome = [int(b) for b in format(choice, f'0{self.n_qubits}b')]
            samples[j, :] = outcome
        return samples

__all__ = ["SamplerQNNEnhanced"]
