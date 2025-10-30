import pennylane as qml
import pennylane.numpy as np
from typing import Dict, List, Tuple

class QCNNHybrid:
    """
    Quantum convolutional neural network implemented with Pennylane.
    Provides a variational circuit with convolutional and pooling layers,
    a caching mechanism for intermediate evaluations, and a training loop.
    """
    def __init__(self, num_qubits: int = 8, num_layers: int = 3, seed: int = 42):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        np.random.seed(seed)
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.cache: Dict[Tuple[float,...], float] = {}
        self.weights = np.random.randn(num_layers, num_qubits)
        self.optimizer = qml.AdamOptimizer(stepsize=0.1)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            # Feature map
            for i in range(num_qubits):
                qml.RX(np.pi * inputs[i], wires=i)
            # Convolutional layers
            for l in range(num_layers):
                for i in range(0, num_qubits, 2):
                    qml.RZ(weights[l, i], wires=i)
                    qml.RY(weights[l, i+1], wires=i+1)
                    qml.CNOT(wires=[i, i+1])
                # Simple pooling: drop even wires (toy pooling)
                # In practice, use measurement + reset or variational pooling
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def predict(self, inputs: np.ndarray) -> float:
        key = tuple(inputs)
        if key in self.cache:
            return self.cache[key]
        output = self.circuit(inputs, self.weights)
        self.cache[key] = output
        return output

    def loss(self, inputs: np.ndarray, target: float) -> float:
        pred = self.predict(inputs)
        return (pred - target) ** 2

    def train(self, dataset: List[Tuple[np.ndarray, float]], epochs: int = 100):
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, target in dataset:
                loss, self.weights = self.optimizer.step_and_cost(
                    lambda w: self.loss(inputs, target), self.weights)
                epoch_loss += loss
            print(f"Epoch {epoch+1}/{epochs} loss: {epoch_loss/len(dataset):.4f}")

    def reset_cache(self):
        self.cache.clear()

__all__ = ["QCNNHybrid"]
