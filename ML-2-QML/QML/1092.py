import pennylane as qml
import numpy as np

class EstimatorQNNHybrid:
    """
    A variational quantum neural network that mirrors the classical
    EstimatorQNNHybrid.  It uses a multi‑layer entangled circuit with
    parameter‑shift gradients and a single Pauli‑Z expectation value
    as the regression output.  The circuit is fully differentiable
    via Pennylane's autograd interface, allowing end‑to‑end training
    with a simple mean‑squared‑error loss.
    """
    def __init__(self, n_qubits: int = 1, layers: int = 2, dev=None):
        self.n_qubits = n_qubits
        self.layers = layers
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)

        # initialise trainable weights
        self.params = np.random.uniform(0, 2 * np.pi, size=(layers, n_qubits))

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            # input encoding
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            # variational layers
            for layer in range(layers):
                for i in range(n_qubits):
                    qml.RZ(weights[layer, i], wires=i)
                # entanglement
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            # measurement
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def __call__(self, inputs: np.ndarray | list[float]) -> float:
        """Evaluate the circuit for a single input vector."""
        return float(self.circuit(inputs, self.params))

    def train(self, X: np.ndarray, y: np.ndarray,
              lr: float = 0.01, epochs: int = 100) -> None:
        """
        Train the QNN with a simple gradient‑descent optimiser.
        The loss is the mean‑squared‑error between predictions and targets.
        """
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for epoch in range(epochs):
            def cost(params):
                preds = np.array([self.circuit(x, params) for x in X])
                return np.mean((preds - y) ** 2)

            self.params = opt.step(cost, self.params)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {cost(self.params):.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Batch inference for a collection of inputs."""
        return np.array([self.circuit(x, self.params) for x in X])

__all__ = ["EstimatorQNNHybrid"]
