import pennylane as qml
import numpy as np

class SamplerQNNEnhanced:
    """
    Variational sampler quantum neural network.
    Architecture:
        • 2 qubits
        • Parameterised Ry rotations on each qubit
        • Two layers of CNOT entanglement
        • Output: probability distribution over computational basis states
    """
    def __init__(self, n_qubits: int = 2, n_layers: int = 2, dev=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = dev or qml.device("default.qubit", wires=self.n_qubits)
        self.weights = self._init_weights()
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _init_weights(self) -> np.ndarray:
        return np.random.uniform(0, 2 * np.pi, size=(self.n_layers, self.n_qubits))

    def _circuit(self, *params):
        params = np.reshape(params, (self.n_layers, self.n_qubits))
        for layer, w in enumerate(params):
            for qubit in range(self.n_qubits):
                qml.RY(w[qubit], wires=qubit)
            if layer < self.n_layers - 1:
                qml.CNOT(wires=[0, 1])
        return qml.probs(wires=range(self.n_qubits))

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the sampler.
        inputs: array of shape (n_qubits,) containing additional rotation angles for the first layer.
        """
        combined = np.concatenate([inputs, self.weights.flatten()])
        return self.qnode(*combined)
