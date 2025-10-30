"""Enhanced fully connected layer with dropout, batch support, and training utilities."""

import numpy as np
import torch
from torch import nn, optim
from typing import Iterable

class FullyConnectedLayer(nn.Module):
    """A versatile fully connected layer supporting dropout, batch inference, and training."""
    def __init__(self, n_features: int = 1, dropout_rate: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.optimizer = None
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional dropout."""
        return torch.tanh(self.linear(self.dropout(x)))

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Mimic the original API: accept parameter list, return expectation."""
        with torch.no_grad():
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            out = self.forward(values)
            return out.mean(dim=0).detach().numpy()

    def train_on(self, X: np.ndarray, y: np.ndarray, epochs: int = 200, lr: float = 1e-3):
        """Simple training loop for regression."""
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            self.optimizer.zero_grad()
            preds = self.forward(X_tensor)
            loss = self.loss_fn(preds, y_tensor)
            loss.backward()
            self.optimizer.step()

    def get_params(self) -> np.ndarray:
        """Return current linear weights as a 1‑D array."""
        return self.linear.weight.detach().cpu().numpy().flatten()

    def set_params(self, params: Iterable[float]) -> None:
        """Load new weights into the linear layer."""
        tensor = torch.as_tensor(list(params), dtype=torch.float32).view(1, -1)
        with torch.no_grad():
            self.linear.weight.copy_(tensor)

def FCL(n_features: int = 1, dropout_rate: float = 0.0):
    """Convenience factory mirroring the original API."""
    return FullyConnectedLayer(n_features=n_features, dropout_rate=dropout_rate)
