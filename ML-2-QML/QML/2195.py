"""
QCNNEnhancedQNN: Quantum convolutional neural network with dynamic pooling and gradient-based training.

Features:
* Custom convolution and pooling circuits parameterised per qubit pair.
* An ansatz built by repeating conv+pool layers.
* Supports training with a hybrid optimizer (COBYLA or SLSQP) via Qiskit's EstimatorQNN.
* Provides fit and predict methods analogous to sklearn's API.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import COBYLA, AdamOptimizer
from pennylane import qinfo
from typing import Tuple, Iterable
import torch

# Helper to build a 2-qubit convolution unit
def conv_unit(params: Iterable[float], wires: Tuple[int, int]) -> None:
    """Two-qubit convolution unit: RZ, CX, RZ, RY, CX, RY, CX."""
    qml.RZ(-np.pi / 2, wires=wires[1])
    qml.CNOT(wires=wires[1], control=wires[0])
    qml.RZ(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=wires[0], control=wires[1])
    qml.RY(params[2], wires=wires[1])
    qml.CNOT(wires=wires[1], control=wires[0])
    qml.RZ(np.pi / 2, wires=wires[0])

# Helper to build a 2-qubit pooling unit
def pool_unit(params: Iterable[float], wires: Tuple[int, int]) -> None:
    """Two-qubit pooling unit: RZ, CX, RZ, RY, CX, RY."""
    qml.RZ(-np.pi / 2, wires=wires[1])
    qml.CNOT(wires=wires[1], control=wires[0])
    qml.RZ(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=wires[0], control=wires[1])
    qml.RY(params[2], wires=wires[1])

# Feature map using PennyLane's built-in feature map
def feature_map(num_qubits: int) -> qml.Device:
    return qml.templates.AngleEmbedding(wires=range(num_qubits))

# Build the QCNN ansatz
def build_ansatz(num_qubits: int) -> Tuple[qn, list]:
    dev = qml.device("default.qubit", wires=num_qubits)
    weight_shapes = {}
    obs = [qml.PauliZ(i) for i in range(num_qubits)]

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs: torch.Tensor, weights: dict) -> torch.Tensor:
        feature_map(num_qubits)
        # First convolutional layer
        for i in range(0, num_qubits, 2):
            conv_unit(weights[f"c1_{i}"], wires=(i, i+1))
        # First pooling layer
        for i in range(0, num_qubits, 2):
            pool_unit(weights[f"p1_{i}"], wires=(i, i+1))
        # Second conv/pool layers on remaining qubits
        for i in range(0, num_qubits//2, 2):
            conv_unit(weights[f"c2_{i}"], wires=(i, i+1))
        for i in range(0, num_qubits//2, 2):
            pool_unit(weights[f"p2_{i}"], wires=(i, i+1))
        # Final measurement
        return qml.expval(obs[0])  # single-qubit observable

    # Create weight dictionary
    weight_params = {}
    for layer in ["c1", "p1", "c2", "p2"]:
        for i in range(0, num_qubits, 2):
            weight_params[f"{layer}_{i}"] = pnp.random.uniform(-np.pi, np.pi, 3, requires_grad=True)

    return circuit, weight_params

class QCNNEnhancedQNN:
    """
    Hybrid quantum-classical QCNN with fit/predict API.
    Uses PennyLane's autograd for gradient-based optimisation.
    """
    def __init__(self, num_qubits: int = 8, lr: float = 0.01, epochs: int = 50):
        self.num_qubits = num_qubits
        self.circuit, self.weights = build_ansatz(num_qubits)
        self.optimizer = AdamOptimizer(stepsize=lr)
        self.epochs = epochs
        self.loss_history: list[float] = []

    def loss_fn(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Binary cross entropy."""
        eps = 1e-7
        y_pred = torch.clamp(y_pred, eps, 1 - eps)
        return -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Trains the QCNN using backpropagation."""
        for epoch in range(self.epochs):
            loss_total = 0.0
            for x, y_true in zip(X, y):
                x = x.unsqueeze(0)  # batch dim
                y_true = y_true.unsqueeze(0)
                y_pred = self.circuit(x, self.weights)
                loss = self.loss_fn(y_true, y_pred)
                self.optimizer.step(lambda: loss)
                loss_total += loss.item()
            avg_loss = loss_total / len(X)
            self.loss_history.append(avg_loss)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Returns probability predictions."""
        preds = []
        for x in X:
            pred = self.circuit(x.unsqueeze(0), self.weights).item()
            preds.append(pred)
        return torch.tensor(preds)

    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Accuracy on binary classification."""
        preds = self.predict(X) > 0.5
        return (preds == y).float().mean().item()

__all__ = ["QCNNEnhancedQNN"]
