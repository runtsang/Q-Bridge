"""Variational quantum classifier with data‑re‑uploading.

The class builds a parameter‑ized circuit, provides a PennyLane QNode that
returns expectation values of Z on each qubit, and exposes a training
routine that uses PennyLane's autograd backend.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]


class QuantumClassifierModel:
    """Variational circuit classifier with data‑re‑uploading and entanglement."""

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        device: str = "default.qubit",
        shots: int = 1000,
    ):
        self.num_qubits = num_qubits
        self.depth = depth
        self.device = device
        self.shots = shots

        self.dev = qml.device(device, wires=num_qubits, shots=shots)
        self.params = torch.randn(num_qubits * depth, requires_grad=True)
        self.encoding = torch.zeros(num_qubits)

        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, *params: np.ndarray) -> List[float]:
        """Data‑re‑uploading ansatz with CZ entanglement."""
        # Encoding
        for i, x in enumerate(self.encoding):
            qml.RX(x, wires=i)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for i in range(self.num_qubits):
                qml.RY(params[idx], wires=i)
                idx += 1
            for i in range(self.num_qubits - 1):
                qml.CZ(i, i + 1)

        # Observables
        return [qml.expval(qml.PauliZ(w)) for w in range(self.num_qubits)]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for a batch of classical data.

        Parameters
        ----------
        X : torch.Tensor
            Shape (batch, num_qubits).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, 2).
        """
        batch_logits = []
        for x in X:
            self.encoding = x.detach().cpu()
            preds = self.qnode(*self.params)
            logits = torch.tensor(preds, dtype=torch.float32)
            batch_logits.append(logits)
        logits = torch.stack(batch_logits)
        # Simple two‑class mapping: use first two qubits as logits
        return logits[:, :2]

    def train_loop(
        self,
        train_loader,
        epochs: int,
        lr: float,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
    ) -> None:
        optimizer = optim.Adam([self.params], lr=lr)
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for X, y in train_loader:
                optimizer.zero_grad()
                logits = self.forward(X)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(train_loader)

    def evaluate(self, loader) -> Tuple[float, int]:
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in loader:
                logits = self.forward(X)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total, total

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(X)
            return torch.argmax(logits, dim=1)


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumClassifierModel, Iterable, Iterable, List[qml.operation.Operator]]:
    """
    Return a tuple that mirrors the classical helper signature.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features.
    depth : int
        Number of variational layers.

    Returns
    -------
    Tuple[QuantumClassifierModel, Iterable, Iterable, List[qml.operation.Operator]]
        The QML model, an encoding mask, a list of parameter sizes, and
        a list of PauliZ observables.
    """
    model = QuantumClassifierModel(num_qubits=num_qubits, depth=depth)
    encoding = list(range(num_qubits))
    param_sizes = [len(model.params)]
    observables = [qml.PauliZ(i) for i in range(num_qubits)]
    return model, encoding, param_sizes, observables
