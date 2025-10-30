import pennylane as qml
import numpy as np

class FCL:
    """
    Variational quantum circuit representing a fully connected layer.
    The circuit depth can be adjusted and the parameters are trainable.
    """
    def __init__(self, n_qubits: int = 1, depth: int = 1,
                 backend: str = "default.qubit", shots: int = 1000):
        self.n_qubits = n_qubits
        self.depth = depth
        self.device = qml.device(backend, wires=n_qubits, shots=shots)
        # Parameters shape: (depth, n_qubits)
        self.params = np.random.uniform(0, 2*np.pi, size=(depth, n_qubits))
        self.qnode = qml.QNode(self._circuit, self.device)

    def _circuit(self, params, x):
        # Input encoding: rotate each qubit by its corresponding input value
        for i in range(self.n_qubits):
            qml.RY(x[i], wires=i)
        # Variational layers
        for d in range(self.depth):
            for i in range(self.n_qubits):
                qml.RZ(params[d, i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
        return qml.expval(qml.PauliZ(0))

    def run(self, thetas: list[float]) -> np.ndarray:
        """
        Evaluate the circuit for each input theta and return expectation values.
        """
        expectations = []
        for theta in thetas:
            x = np.zeros(self.n_qubits)
            x[0] = theta
            expval = self.qnode(self.params, x)
            expectations.append(expval)
        return np.array(expectations)

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 100, lr: float = 0.01):
        """
        Train the variational parameters using gradient descent on MSE loss.
        """
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for _ in range(epochs):
            def cost(params):
                preds = []
                for x in X:
                    x_vec = np.zeros(self.n_qubits)
                    x_vec[0] = x
                    preds.append(self.qnode(params, x_vec))
                preds = np.array(preds)
                return np.mean((preds - y)**2)
            self.params = opt.step(cost, self.params)

__all__ = ["FCL"]
