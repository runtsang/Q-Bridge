"""Enhanced classical feed‑forward regressor with regularisation and training helper."""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

class EstimatorQNN(nn.Module):
    """A 3‑layer MLP with batch‑norm, dropout and residual skip.

    * Input dimension: 2
    * Hidden layers: 8 → 4
    * Dropout probability: 0.2
    * Residual connection from first to second layer
    * Supports a quick ``train`` class‑method for supervised regression.
    """
    def __init__(self, input_dim: int = 2, hidden1: int = 8, hidden2: int = 4,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1)
        )
        # Residual connection weights (learnable)
        self.res = nn.Linear(input_dim, hidden2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.net[0:4](x)          # first linear + bn + tanh + dropout
        h2 = self.net[4:8](h1)         # second linear + bn + tanh + dropout
        h2 = h2 + self.res(x)          # residual
        return self.net[8](h2)         # output layer

    @classmethod
    def train(cls,
              X: torch.Tensor,
              y: torch.Tensor,
              epochs: int = 200,
              batch_size: int = 32,
              lr: float = 1e-3,
              device: str | None = None) -> "EstimatorQNN":
        """Convenience training routine returning a fitted model."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = cls().to(device)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
        return model

__all__ = ["EstimatorQNN"]
