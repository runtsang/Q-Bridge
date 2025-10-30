import pennylane as qml
import pennylane.numpy as np

class EstimatorQNNGen391:
    """Variational quantum circuit with a 4‑qubit feature map and trainable rotation angles."""
    def __init__(self, num_qubits: int = 4, device_name: str = "default.qubit", shots: int = 1024) -> None:
        self.num_qubits = num_qubits
        self.wires = list(range(num_qubits))
        self.dev = qml.device(device_name, wires=self.wires, shots=shots)
        self._build_qnode()

    def _feature_map(self, inputs: np.ndarray) -> None:
        for i, w in enumerate(self.wires):
            qml.RY(inputs[i], wires=w)
        for i in range(self.num_qubits - 1):
            qml.CNOT(self.wires[i], self.wires[i + 1])

    def _variational_layer(self, params: np.ndarray) -> None:
        for i, w in enumerate(self.wires):
            qml.RX(params[i], wires=w)
        for i in range(self.num_qubits - 1):
            qml.CNOT(self.wires[i], self.wires[i + 1])

    def _circuit(self, inputs: np.ndarray, params: np.ndarray) -> list[np.ndarray]:
        self._feature_map(inputs)
        self._variational_layer(params)
        return [qml.expval(qml.PauliZ(w)) for w in self.wires]

    def _build_qnode(self) -> None:
        @qml.qnode(self.dev, interface="autograd")
        def qnode(inputs: np.ndarray, params: np.ndarray) -> list[np.ndarray]:
            return self._circuit(inputs, params)

        self.qnode = qnode
        self.loss = qml.loss.mean_squared_error
        self.optimizer = qml.GradientDescentOptimizer(stepsize=0.1)

    def predict(self, inputs: np.ndarray, params: np.ndarray) -> float:
        return float(np.mean(self.qnode(inputs, params)))

    def train(self, data: np.ndarray, targets: np.ndarray, epochs: int = 100, batch_size: int = 16) -> np.ndarray:
        params = 0.01 * np.random.randn(self.num_qubits)
        for _ in range(epochs):
            for i in range(0, len(data), batch_size):
                batch_x = data[i:i+batch_size]
                batch_y = targets[i:i+batch_size]
                grads = self.optimizer.gradient(
                    lambda p: self.loss(self.qnode(batch_x, p), batch_y),
                    params)
                params = self.optimizer.apply_gradients(params, grads)
        return params

def EstimatorQNN() -> EstimatorQNNGen391:
    """Factory that returns an instance of the extended quantum estimator."""
    return EstimatorQNNGen391()

__all__ = ["EstimatorQNNGen391", "EstimatorQNN"]
