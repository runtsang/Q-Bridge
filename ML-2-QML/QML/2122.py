from __future__ import annotations

import pennylane as qml
import pennylane.numpy as np
import torch
from typing import Iterable, Tuple, List

__all__ = ["build_classifier_circuit", "QuantumClassifierModel"]


class SimpleSGD:
    """A minimal SGD optimizer for dict‑based parameters."""

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, params: dict, grads: dict):
        for k in params:
            params[k] -= self.lr * grads[k]


class QuantumClassifierModel:
    """
    Variational quantum classifier that exposes the same public API as the classical counterpart.
    It can be trained with a hybrid loss (cross‑entropy or MSE) using Pennylane's autograd.
    """

    def __init__(self, num_qubits: int, depth: int, dropout: float = 0.0,
                 loss_type: str = "cross_entropy"):
        self.num_qubits = num_qubits
        self.depth = depth
        self.dropout = dropout
        self.loss_type = loss_type

        # Quantum device
        self.dev = qml.device("default.qubit", wires=num_qubits)

        # Parameter shapes for each variational layer
        self.weight_shapes = {f"w_{d}": (num_qubits,) for d in range(depth)}
        self.weights = {name: np.random.randn(*shape) for name, shape in self.weight_shapes.items()}

        # Metadata
        self.encoding = list(range(num_qubits))
        self.weight_sizes = [np.prod(shape) for shape in self.weight_shapes.values()]
        self.observables = [qml.PauliZ(i) for i in range(num_qubits)]

    def _ansatz(self, x: np.ndarray, weights: dict):
        for i, qubit in enumerate(self.encoding):
            qml.RX(x[i], wires=qubit)

        for d in range(self.depth):
            for qubit in self.encoding:
                qml.RY(weights[f"w_{d}"][qubit], wires=qubit)
            for qubit in range(self.num_qubits - 1):
                qml.CZ(self.encoding[qubit], self.encoding[qubit + 1])

    @qml.qnode
    def circuit(self, x: np.ndarray, weights: dict) -> List[float]:
        self._ansatz(x, weights)
        return [qml.expval(obs) for obs in self.observables]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to numpy for Pennylane
        x_np = x.detach().cpu().numpy()
        expvals = self.circuit(x_np, self.weights)
        return torch.tensor(expvals, dtype=torch.float32)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "cross_entropy":
            logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
            log_probs = logits - torch.log(torch.sum(torch.exp(logits), dim=1, keepdim=True))
            return -torch.mean(log_probs[range(labels.size(0)), labels])
        elif self.loss_type == "mse":
            return torch.mean((logits - torch.nn.functional.one_hot(labels, num_classes=2).float()) ** 2)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                   optimizer: SimpleSGD) -> torch.Tensor:
        x, y = batch

        def loss_fn(params):
            expvals = self.circuit(x.detach().cpu().numpy(), params)
            logits = torch.tensor(expvals, dtype=torch.float32)
            return self.compute_loss(logits, y)

        loss = loss_fn(self.weights)
        grads = qml.grad(loss_fn)(self.weights)
        optimizer.step(self.weights, grads)
        return loss


def build_classifier_circuit(num_qubits: int, depth: int,
                             dropout: float = 0.0,
                             loss_type: str = "cross_entropy") -> Tuple[qml.QNode, Iterable, Iterable, List[qml.PauliZ]]:
    """
    Construct a variational circuit and return it along with metadata that
    mirrors the classical helper's interface.
    """
    model = QuantumClassifierModel(num_qubits, depth, dropout, loss_type)
    return model.circuit, model.encoding, model.weight_sizes, model.observables
