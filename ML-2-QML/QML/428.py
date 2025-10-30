import pennylane as qml
import pennylane.numpy as np
from pennylane.optimize import AdamOptimizer
from pennylane.losses import MSELoss
from typing import List

class QCNNModel:
    """
    Quantum Convolutional Neural Network built with Pennylane.
    Implements a Z‑feature map followed by a stack of strongly entangling layers.
    Parameter sharing across layers reduces the trainable space and mimics
    translational invariance. The network outputs a single scalar via
    measurement of the first qubit, then passed through a sigmoid.
    """
    def __init__(
        self,
        num_qubits: int = 8,
        layers: int = 3,
        shared_params: bool = True,
        seed: int = 42,
    ) -> None:
        self.num_qubits = num_qubits
        self.layers = layers
        self.shared_params = shared_params
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self._build_ansatz()
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _build_ansatz(self) -> None:
        self.feature_map = qml.feature_maps.ZFeatureMap(
            num_qubits=self.num_qubits, reps=1
        )
        self.ansatz = qml.StronglyEntanglingLayers(
            num_layers=self.layers,
            wires=range(self.num_qubits),
            seed=0,
        )
        self.params = self.ansatz.parameters
        if self.shared_params:
            shared = self.params[: len(self.params) // self.layers]
            self.params = np.tile(shared, (self.layers, 1)).flatten()

    def _circuit(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        self.feature_map(x)
        self.ansatz(weights)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.qnode(x, self.params)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        lr: float = 0.01,
    ) -> None:
        opt = AdamOptimizer(lr)
        loss = MSELoss()
        for _ in range(epochs):
            def cost(weights):
                preds = self.qnode(X, weights)
                preds = np.mean(preds, axis=1)  # simple pooling
                return loss(preds, y)
            self.params = opt.step(cost, self.params)

def QCNN() -> QCNNModel:
    """
    Factory returning a configured QCNNModel instance.
    """
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
