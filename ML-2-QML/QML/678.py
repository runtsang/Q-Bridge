"""Fraud detection using a PennyLane variational circuit with a classical post‑processing layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import pennylane as qml
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class FraudLayerParameters:
    """Parameters for a variational layer (used only for documentation)."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(v: float, bound: float) -> float:
    return max(-bound, min(bound, v))


class FraudDetector(nn.Module):
    """Hybrid quantum‑classical fraud detector.

    Uses a PennyLane QNode that outputs a 2‑dimensional feature vector;
    a classical linear layer maps this to a single logit.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.device = qml.device("default.qubit", wires=2)
        self.input_params = input_params
        self.layers = layers
        self.qnode = qml.QNode(self._circuit, self.device, interface="torch")
        self.classifier = nn.Linear(2, 1)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self._optimizer = optim.Adam(list(self.parameters()), lr=self.lr)
        self._criterion = nn.BCEWithLogitsLoss()

    def _circuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Variational circuit producing a 2‑dim output."""
        # Encode input
        qml.RX(x[0], wires=0)
        qml.RY(x[1], wires=1)

        # Apply layers
        for i, p in enumerate(params):
            # Simple parameterised two‑qubit entangler
            qml.CRX(p[0], wires=[0, 1])
            qml.CRY(p[1], wires=[0, 1])
            qml.CZ(wires=[0, 1])
            # Per‑qubit rotations
            qml.RX(p[2], wires=0)
            qml.RY(p[3], wires=1)

        # Measurement
        return torch.stack([qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits."""
        # Gather parameters into a tensor
        param_list = [torch.tensor([p.bs_theta, p.bs_phi, p.phases[0], p.phases[1]], dtype=torch.float32)
                      for p in self.layers]
        params = torch.stack(param_list)
        features = self.qnode(x, params)
        return self.classifier(features).squeeze(-1)

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            self._optimizer.zero_grad()
            logits = self(xb)
            loss = self._criterion(logits, yb.float())
            loss.backward()
            self._optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        return epoch_loss / len(loader.dataset)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        best_loss = float("inf")
        patience = 10
        counter = 0
        for epoch in range(self.epochs):
            loss = self._train_one_epoch(loader)
            if loss < best_loss:
                best_loss = loss
                counter = 0
            else:
                counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self(X)
            probs = torch.sigmoid(logits)
            return (probs > 0.5).long()

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        self.eval()
        with torch.no_grad():
            logits = self(X)
            loss = self._criterion(logits, y.float()).item()
            preds = (torch.sigmoid(logits) > 0.5).long()
            acc = (preds == y).float().mean().item()
        return {"loss": loss, "accuracy": acc}


__all__ = ["FraudLayerParameters", "FraudDetector"]
