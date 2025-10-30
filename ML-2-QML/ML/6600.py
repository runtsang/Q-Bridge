import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class FCL(nn.Module):
    """
    Classical fully‑connected layer with linear, batch‑norm, dropout.
    Provides `forward`, `train_model`, and `run` methods.
    """
    def __init__(self, n_features: int = 1, dropout: float = 0.2):
        super().__init__()
        self.n_features = n_features
        self.linear = nn.Linear(n_features, 1)
        self.bn = nn.BatchNorm1d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.bn(out)
        out = F.tanh(out)
        out = self.dropout(out)
        return out

    def train_model(self, data: np.ndarray, targets: np.ndarray, lr=0.01, epochs=200):
        """
        Train the layer using MSE loss and Adam optimizer.
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        X = torch.from_numpy(data).float()
        y = torch.from_numpy(targets).float().unsqueeze(1)
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = self.forward(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Return the output of the layer for a 1‑D array of theta values.
        """
        with torch.no_grad():
            inp = torch.from_numpy(thetas).float().unsqueeze(1)
            out = self.forward(inp).cpu().numpy()
        return out.squeeze()

__all__ = ["FCL"]
