"""Quantum neural network estimator using PennyLane.

Features:
  • 2‑qubit variational circuit with angle encoding and
    StronglyEntanglingLayers ansatz.
  • sklearn‑style `fit`, `predict`, and `score` methods.
  • Autograd via PyTorch interface.
  • Early‑stopping and configurable training hyperparameters.
  • Supports hybrid classical‑quantum training loops.
"""

import pennylane as qml
import torch
from torch import nn
import numpy as np
from sklearn.metrics import r2_score


# Device: use the default qubit simulator; can be swapped to a real backend.
dev = qml.device("default.qubit", wires=2)


def quantum_circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Angle‑encoded 2‑qubit circuit with a variational layer."""
    # Input encoding
    qml.RY(inputs[0], wires=0)
    qml.RY(inputs[1], wires=1)

    # Variational ansatz
    qml.StronglyEntanglingLayers(weights, wires=[0, 1])

    # Measurement
    return qml.expval(qml.PauliZ(0))


class EstimatorQNN(nn.Module):
    """Quantum neural network regression model."""
    def __init__(
        self,
        input_dim: int = 2,
        num_layers: int = 2,
        lr: float = 1e-3,
        epochs: int = 2000,
        batch_size: int = 32,
        early_stop_patience: int = 20,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop_patience = early_stop_patience

        # Weight shape for StronglyEntanglingLayers: (num_layers, 3, num_wires)
        self.weight_shapes = (num_layers, 3, 2)
        self.weight_params = nn.Parameter(torch.randn(self.weight_shapes))
        self.optimizer = torch.optim.Adam([self.weight_params], lr=lr)
        self.criterion = nn.MSELoss()

        self.qnode = qml.QNode(quantum_circuit, dev, interface="torch")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.to(self.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute predictions for a batch of inputs."""
        # inputs: (batch, input_dim)
        preds = torch.stack([self.qnode(x, self.weight_params) for x in inputs])
        return preds.unsqueeze(1)

    # ------------------------------------------------------------------
    #  Training utilities
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "EstimatorQNN":
        """Fit the quantum model on the supplied data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray, shape (n_samples,)
            Target values.
        """
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        best_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                preds = self(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(dataset)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.early_stop_patience:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for the supplied data."""
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.eval()
        with torch.no_grad():
            preds = self(X).cpu().numpy().flatten()
        return preds

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R² score for the predictions."""
        preds = self.predict(X)
        return r2_score(y, preds)


__all__ = ["EstimatorQNN"]
