import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Iterable

class FCL(nn.Module):
    """
    A multi‑layer fully connected neural network with optional dropout and batch normalization.
    Provides `train_on` and `evaluate` helpers for quick prototyping.
    """
    def __init__(self, input_dim: int, hidden_dims: Iterable[int] = (64, 32), output_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def train_on(self, data_loader: DataLoader, epochs: int = 10, lr: float = 1e-3, loss_fn=nn.MSELoss()):
        """
        Train the network on the provided DataLoader.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for _ in range(epochs):
            for batch_x, batch_y in data_loader:
                optimizer.zero_grad()
                pred = self(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()

    def evaluate(self, data_loader: DataLoader, loss_fn=nn.MSELoss()):
        """
        Evaluate the network on the provided DataLoader and return average loss.
        """
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                pred = self(batch_x)
                total_loss += loss_fn(pred, batch_y).item()
        return total_loss / len(data_loader)

__all__ = ["FCL"]
