import pennylane as qml
import numpy as np

class SamplerQNN:
    """
    Variational quantum sampler network built with Pennylane.
    Supports multiple entangling layers and sampling from the circuit.
    """
    def __init__(self, n_qubits: int = 2, n_layers: int = 2,
                 device_name: str = "default.qubit", seed: int | None = None) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(device_name, wires=n_qubits)
        if seed is not None:
            np.random.seed(seed)
        # Initialize random weights: shape (n_layers, n_qubits, 3) for RZ,RY,RZ rotations
        self.weights = np.random.randn(n_layers, n_qubits, 3)
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, input_angles: np.ndarray, weights: np.ndarray) -> list[float]:
        # Input rotations
        for i in range(self.n_qubits):
            qml.RY(input_angles[i], wires=i)
        # Entangling layers
        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                qml.RZ(weights[l][q][0], wires=q)
                qml.RY(weights[l][q][1], wires=q)
                qml.RZ(weights[l][q][2], wires=q)
            # Entangle adjacent qubits
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        # Return expectation values of PauliZ as a proxy for probabilities
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Execute the circuit and return probabilities derived from expectation values.
        """
        expvals = self.qnode(inputs, self.weights)
        # Convert expectation values [-1,1] to probabilities [0,1]
        probs = (np.array(expvals) + 1) / 2
        return probs

    def sample(self, inputs: np.ndarray, n_shots: int = 1024) -> dict:
        """
        Sample measurement outcomes from the circuit.
        Returns a dictionary of bitstring counts.
        """
        sampler = qml.Sampler(self._circuit, shots=n_shots)
        result = sampler(inputs, self.weights)
        return result.counts

__all__ = ["SamplerQNN"]
