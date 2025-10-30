import pennylane as qml
from pennylane import numpy as np
from pennylane_qnn import SamplerQNN as QSamplerQNN

class SamplerQNN:
    """Quantum sampler network with a variational circuit and classical readout."""
    def __init__(self, n_qubits: int = 2, n_layers: int = 2) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        # Trainable weights for the variational circuit
        self.weights = np.random.randn(n_layers, n_qubits)

    def circuit(self, inputs, weights):
        """Parameterized quantum circuit."""
        for i in range(self.n_layers):
            qml.RY(inputs[i], wires=i)
            qml.CNOT(wires=[i, (i+1)%self.n_qubits])
            qml.RY(weights[i], wires=i)
        return qml.expval(qml.PauliZ(0))

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Return sampled probabilities for each input."""
        probs = []
        for inp in inputs:
            prob = qml.QNode(self.circuit, self.dev)(inp, self.weights)
            probs.append(prob)
        return np.array(probs)
