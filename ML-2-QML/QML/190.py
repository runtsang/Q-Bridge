import pennylane as qml
import pennylane.numpy as np
from pennylane.optimize import AdamOptimizer
from pennylane.devices import DefaultQubit

class EstimatorQNN:
    """
    Variational quantum regressor with feature‑map and ansatz.

    Parameters
    ----------
    circuit : qml.QNode
        The variational circuit that maps inputs to measurement outcomes.
    observable : qml.operation.Observable
        Observable whose expectation value is returned as the prediction.
    optimizer : pennylane.optimize.Optimizer, optional
        Optimiser used for training the weight parameters.
    """
    def __init__(
        self,
        circuit: qml.QNode,
        observable: qml.operation.Observable,
        optimizer: qml.optimize.Optimizer | None = None,
    ):
        self.circuit = circuit
        self.observable = observable
        self.optimizer = optimizer or AdamOptimizer(0.01)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return expectation value for each data point."""
        preds = []
        for x in data:
            preds.append(np.real(self.circuit(*x)))
        return np.array(preds)

    def loss(self, data: np.ndarray, target: np.ndarray) -> float:
        preds = self.predict(data)
        return np.mean((preds - target) ** 2)

    def train(self, data: np.ndarray, target: np.ndarray, epochs: int = 200):
        """Simple training loop that optimises weight_params."""
        for _ in range(epochs):
            grads = self.optimizer.gradient(lambda w: self.loss(data, target), self.circuit.parameters)
            self.optimizer.step(grads, self.circuit.parameters)

def EstimatorQNN() -> EstimatorQNN:
    """Convenience constructor mirroring the original API."""
    dev = DefaultQubit()

    @qml.qnode(dev)
    def circuit(x1, x2):
        qml.Hadamard(0)
        qml.RY(x1, 0)
        qml.RX(x2, 0)
        return qml.expval(qml.PauliY(0))

    observable = qml.PauliY(0)
    return EstimatorQNN(circuit, observable)

__all__ = ["EstimatorQNN"]
