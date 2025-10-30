"""
QuantumClassifierModel (quantum) – Pennylane implementation with hybrid training and parameter‑shift gradients.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from typing import Iterable, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim


class QuantumClassifierModel:
    """
    A hybrid quantum‑classical classifier that builds a variational circuit and trains it with
    gradient‑shift rules. The interface mirrors the classical counterpart for easy swapping.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int = 2,
        device: str = "cpu",
        lr: float = 0.01,
        epochs: int = 200,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        num_qubits : int
            Number of qubits / input features.
        depth : int, default 2
            Depth of the variational ansatz.
        device : str, default "cpu"
            Pennylane device name: 'default.qubit' or 'lightning.qubit' etc.
        lr : float, default 0.01
            Learning rate for Adam optimizer.
        epochs : int, default 200
            Number of training epochs.
        seed : int | None, default None
            Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.num_qubits = num_qubits
        self.depth = depth
        self.device_name = device
        self.lr = lr
        self.epochs = epochs

        # PennyLane device
        self.dev = qml.device(self.device_name, wires=self.num_qubits)

        # Parameter initialization
        self.params = np.random.randn(self.depth * self.num_qubits)
        self.obs = [qml.PauliZ(i) for i in range(self.num_qubits)]

        # Classical post‑processing head (simple linear layer)
        self.classifier = nn.Linear(self.num_qubits, 2)
        self.classifier.apply(self._init_weights)

        self.optimizer = optim.Adam(
            list(self.classifier.parameters()) + [{"params": self.params}],
            lr=self.lr,
        )
        self.criterion = nn.CrossEntropyLoss()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

    def circuit(self, x, params):
        """
        Variational circuit with data encoding and depth‑wise ansatz.
        """
        for i, val in enumerate(x):
            qml.RX(val, wires=i)

        idx = 0
        for _ in range(self.depth):
            for i in range(self.num_qubits):
                qml.RY(params[idx], wires=i)
                idx += 1
            for i in range(self.num_qubits - 1):
                qml.CZ(wires=[i, i + 1])

        return [qml.expval(o) for o in self.obs]

    def _qml_predict(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the quantum circuit and return expectation values.
        """
        return self.dev.batch_call(lambda x_: self.circuit(x_, self.params))(x)

    def train(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> List[float]:
        """
        Hybrid training loop: quantum forward pass + classical linear head.
        """
        losses: List[float] = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            # Shuffle data
            idx = np.random.permutation(len(X))
            X_shuffled, y_shuffled = X[idx], y[idx]

            for xb, yb in zip(X_shuffled, y_shuffled):
                # Quantum forward
                q_out = self._qml_predict(xb[np.newaxis, :])[0]  # shape (num_qubits,)
                q_out = torch.tensor(q_out, dtype=torch.float32, device=self.dev.device)

                # Classical head
                logits = self.classifier(q_out)
                loss = self.criterion(logits.unsqueeze(0), torch.tensor([yb], dtype=torch.long))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(X)
            losses.append(epoch_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{self.epochs} – loss: {epoch_loss:.4f}")

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return class predictions for a batch of inputs.
        """
        preds = []
        for xb in X:
            q_out = self._qml_predict(xb[np.newaxis, :])[0]
            q_out = torch.tensor(q_out, dtype=torch.float32, device=self.dev.device)
            logits = self.classifier(q_out)
            preds.append(int(logits.argmax().item()))
        return np.array(preds)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Return accuracy and loss on a test set.
        """
        correct = 0
        total = len(X)
        losses = []
        for xb, yb in zip(X, y):
            q_out = self._qml_predict(xb[np.newaxis, :])[0]
            q_out = torch.tensor(q_out, dtype=torch.float32, device=self.dev.device)
            logits = self.classifier(q_out)
            loss = self.criterion(logits.unsqueeze(0), torch.tensor([yb], dtype=torch.long))
            losses.append(loss.item())
            pred = logits.argmax().item()
            if pred == yb:
                correct += 1
        acc = correct / total
        return acc, np.mean(losses)

    @property
    def metadata(self) -> dict:
        """
        Return metadata: parameter count, circuit depth, and observable list.
        """
        return {
            "num_qubits": self.num_qubits,
            "depth": self.depth,
            "params_shape": self.params.shape,
            "observables": [str(o) for o in self.obs],
        }

__all__ = ["QuantumClassifierModel"]
