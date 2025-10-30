import pennylane as qml
import numpy as np

class FCL:
    """
    Variational quantum circuit acting as a fully connected layer.
    Uses a single qubit with a parameterized Ry gate followed by measurement.
    Provides batched forward evaluation and a simple gradient‑descent training loop.
    """
    def __init__(self, n_qubits: int = 1, dev: qml.Device = None):
        self.n_qubits = n_qubits
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            qml.Hadamard(wires=range(n_qubits))
            for i, p in enumerate(params):
                qml.RY(p, wires=i)
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, thetas: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit for a batch of parameter vectors.
        """
        return np.array([self.circuit(t) for t in thetas])

    def train(self, data: np.ndarray, labels: np.ndarray, lr: float = 0.01, epochs: int = 100):
        """
        Gradient‑descent training to fit the circuit to the provided data.
        """
        for _ in range(epochs):
            grads = np.zeros_like(data[0])
            for t, y in zip(data, labels):
                pred = self.circuit(t)
                loss = (pred - y) ** 2
                grads += qml.grad(self.circuit)(t) * 2 * (pred - y)
            self.circuit = qml.qnode(self.dev, interface="autograd")(lambda params: self.circuit(params))
            self.circuit = lambda params: self.circuit(params)  # update with new parameters
            self.circuit.params = self.circuit.params - lr * grads / len(data)

__all__ = ["FCL"]
