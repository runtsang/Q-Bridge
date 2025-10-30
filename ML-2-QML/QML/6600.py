import pennylane as qml
import numpy as np

class FCL:
    """
    Variational quantum circuit that mimics a fully‑connected layer.
    Provides `forward`, `train_model`, and `run` methods.
    """
    def __init__(self, n_qubits=1, dev=None):
        self.n_qubits = n_qubits
        self.dev = dev or qml.device("default.qubit", wires=n_qubits, shots=1024)
        # Parameter matrix of shape (n_qubits, 1)
        self.params = np.random.randn(n_qubits, 1)

    def circuit(self, thetas, params=None):
        params = params if params is not None else self.params
        for i in range(self.n_qubits):
            qml.RY(thetas[i], wires=i)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    def forward(self, thetas):
        return self.circuit(thetas)

    def train_model(self, data, targets, lr=0.01, epochs=200):
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for _ in range(epochs):
            def cost(params):
                preds = np.array([self.circuit(t, params) for t in data])
                return np.mean((preds - targets) ** 2)
            self.params = opt.step(cost, self.params)

    def run(self, thetas):
        return np.array([self.forward(t) for t in thetas]).reshape(-1, 1)

__all__ = ["FCL"]
