import numpy as np
import pennylane as qml

class EstimatorQNN:
    """Variational quantum circuit regressor using Pennylane."""
    def __init__(self, num_qubits: int = 4, num_layers: int = 2, device_name: str = "default.qubit"):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device(device_name, wires=num_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")
        self.params = np.random.randn(num_layers, num_qubits, 3)  # rotation angles

    def _circuit(self, inputs: np.ndarray, params: np.ndarray):
        # Angle encoding
        for i, x in enumerate(inputs):
            qml.RX(x, wires=i)
        # Variational layers
        for layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                qml.RY(params[layer, qubit, 0], wires=qubit)
                qml.RZ(params[layer, qubit, 1], wires=qubit)
                qml.RX(params[layer, qubit, 2], wires=qubit)
            # Entangling
            for qubit in range(self.num_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
        # Measurement
        return qml.expval(qml.PauliZ(0))

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return expectation value for each input vector."""
        return np.array([self.qnode(inp, self.params) for inp in inputs])

    def loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Mean squared error."""
        return np.mean((predictions - targets) ** 2)

    def train(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 200):
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for epoch in range(epochs):
            preds = self.predict(X)
            loss_val = self.loss(preds, y)
            grads = qml.grad(lambda p: self.loss(self.predict(X), y))(self.params)
            self.params = opt.step(grads, self.params)
