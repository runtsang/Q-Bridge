"""Quantum variational classifier that mirrors the classical helper interface with a full training pipeline."""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class QuantumClassifier:
    """
    Variational quantum classifier that mirrors the classical helper interface.
    Uses a data‑re‑uploading ansatz with parameter‑shift gradient and supports
    hybrid training (classical optimizer on quantum parameters).
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int = 2,
        device: str = "default.qubit",
        shots: int = 1000,
        lr: float = 1e-3,
        seed: int | None = None,
    ):
        self.num_qubits = num_qubits
        self.depth = depth
        self.device = device
        self.shots = shots
        self.lr = lr
        self.seed = seed

        self.dev = qml.device(device, wires=num_qubits, shots=shots, seed=seed)
        self._build_circuit()
        self.params = self.circuit.parameters
        self.optimizer = optim.Adam(
            [torch.tensor(p, requires_grad=True) for p in self.params], lr=lr
        )

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor, theta: torch.Tensor):
            # Data encoding
            for i, wire in enumerate(range(self.num_qubits)):
                qml.RX(x[i], wires=wire)

            # Parameterised layers
            idx = 0
            for _ in range(self.depth):
                for wire in range(self.num_qubits):
                    qml.RY(theta[idx], wires=wire)
                    idx += 1
                for wire in range(self.num_qubits - 1):
                    qml.CZ(wires=[wire, wire + 1])

            # Measurement: expectation of Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.circuit = circuit

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Map a batch of samples to class logits via a linear readout."""
        probs = []
        for x in X:
            out = self.circuit(x, torch.tensor(self.params))
            probs.append(out)
        probs = torch.stack(probs)
        # Linear readout: sum of qubit expectations
        logits = torch.matmul(probs, torch.ones(self.num_qubits, 2))
        return logits

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                logits = self.forward(xb)
                loss = criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(loader.dataset)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} loss={epoch_loss:.4f}")

    def predict(self, X: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(X)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            return (probs >= threshold).long()

    def train(self):
        self.circuit.train()

    def eval(self):
        self.circuit.eval()


__all__ = ["QuantumClassifier"]
