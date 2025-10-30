"""Enhanced classical regression model with dropout, layer‑norm, and early stopping."""
from __future__ import annotations

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

class EstimatorNN(nn.Module):
    """Feed‑forward network with residual‑style layers and regularisation."""
    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | tuple[int,...] = (16, 8), dropout: float = 0.1) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def train_on(
        self,
        dataloader: DataLoader,
        epochs: int = 50,
        lr: float = 1e-3,
        patience: int = 10,
        device: str = "cpu",
    ) -> None:
        """Simple early‑stopping training loop."""
        self.to(device)
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        best_loss = float("inf")
        bad_epochs = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in dataloader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = self(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(dataloader.dataset)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

def EstimatorQNN() -> EstimatorNN:
    """Return an initialized estimator."""
    return EstimatorNN()

__all__ = ["EstimatorQNN"]
