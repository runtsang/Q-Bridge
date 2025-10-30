import pennylane as qml
from pennylane import numpy as np

class SamplerQNN:
    """
    Variational sampler implemented with PennyLane.
    2‑qubit circuit with parameterized rotations and entangling gates.
    """
    def __init__(self, n_qubits: int = 2, n_layers: int = 3, seed: int | None = None) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        if seed is not None:
            np.random.seed(seed)
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.weights = np.random.uniform(0, 2*np.pi, (n_layers, n_qubits))
        self.params = np.random.uniform(0, 2*np.pi, n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params, weights):
            # input rotations
            for i in range(n_qubits):
                qml.RY(params[i], wires=i)
            # variational layers
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                qml.CNOT(wires=[0, 1])
            return qml.probs(wires=range(n_qubits))

        self.circuit = circuit

    def sample(self, n: int = 1) -> np.ndarray:
        """
        Draw samples from the quantum circuit.
        """
        probs = self.circuit(self.params, self.weights)
        return np.random.choice(len(probs), size=n, p=probs)

    def get_params(self):
        return {"params": self.params, "weights": self.weights}

__all__ = ["SamplerQNN"]
