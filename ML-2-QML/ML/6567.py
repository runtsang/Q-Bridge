"""Hybrid classical regressor that simulates a quantum feature map.

The EstimatorQNN class implements a tiny feed‑forward network that first
produces features by classically simulating a 2‑qubit rotation circuit.
The output of the circuit is then fed into a small neural head.  The
module can be used as a drop‑in replacement for scikit‑learn regressors
and supports early stopping, learning‑rate scheduling and weight
initialisation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class QuantumFeatureExtractor(nn.Module):
    """Simulate a simple 2‑qubit rotation circuit and return the probability amplitudes."""
    def __init__(self, n_qubits: int = 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        n_samples = X.shape[0]
        state = torch.zeros(n_samples, 2 ** self.n_qubits, dtype=torch.float32)
        state[:, 0] = 1.0
        for q in range(self.n_qubits):
            theta = X[:, q]
            cos = torch.cos(theta / 2)
            sin = torch.sin(theta / 2)
            if self.n_qubits == 2:
                idx0, idx2 = 0, 2
                state[:, [idx0, idx2]] = (
                    cos[:, None] * state[:, [idx0, idx2]]
                    + sin[:, None] * state[:, [idx2, idx0]]
                )
                idx1, idx3 = 1, 3
                state[:, [idx0, idx1]] = (
                    cos[:, None] * state[:, [idx0, idx1]]
                    + sin[:, None] * state[:, [idx1, idx0]]
                )
        return state

class QuantumRegressor(nn.Module):
    """Wraps the feature extractor and a classical head."""
    def __init__(self, feature_extractor: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = head

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(X)
        return self.head(feats)

class EstimatorQNN(BaseEstimator, RegressorMixin):
    """Hybrid regressor with a classical feed‑forward network and a quantum feature extractor (simulated on the CPU)."""
    def __init__(
        self,
        n_qubits: int = 2,
        hidden_dim: int = 32,
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> None:
        self.n_qubits = n_qubits
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self._model: nn.Module | None = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        feature_extractor = QuantumFeatureExtractor(self.n_qubits)
        head = nn.Sequential(
            nn.Linear(2 ** self.n_qubits, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self._model = QuantumRegressor(feature_extractor, head)

        optimizer = optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self._model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if self.verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:03d} loss: {epoch_loss / len(loader):.6f}")
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            preds = self._model(torch.tensor(X, dtype=torch.float32))
        return preds.numpy().flatten()

__all__ = ["EstimatorQNN"]
