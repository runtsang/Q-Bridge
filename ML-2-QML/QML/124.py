import pennylane as qml
import pennylane.numpy as np

class SamplerQNN:
    """
    Variational quantum sampler network.
    • 2‑qubit circuit with parameterized Ry rotations and CNOT entanglement.
    • Configurable number of layers and device.
    • Provides both state‑vector probabilities and sampling functionality.
    """
    def __init__(self, dev=None, n_layers: int = 2, seed: int = 42):
        self.n_qubits = 2
        self.n_layers = n_layers
        self.dev = dev or qml.device("default.qubit", wires=self.n_qubits)
        np.random.seed(seed)
        self.params = np.random.uniform(0, 2*np.pi, (n_layers, self.n_qubits))
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

    def _circuit(self, params, x):
        # Input encoding: Ry rotations
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)
        # Variational layers
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                qml.RY(params[layer, qubit], wires=qubit)
            qml.CNOT(wires=[0, 1])
        # Return samples of all qubits
        return qml.sample()

    def probability_distribution(self, x):
        """Return probability distribution over computational basis."""
        sv = qml.statevector(self._circuit, self.params, x)
        probs = np.abs(sv)**2
        return probs

    def sample(self, x, n_shots: int = 1000):
        """Sample measurement outcomes from the circuit."""
        return self.qnode(self.params, x, shots=n_shots)
