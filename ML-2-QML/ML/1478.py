"""Enhanced fully connected layer with training capabilities.

This module implements a small multi‑layer perceptron (MLP) that can be
trained on tabular data.  It keeps the simple ``run`` interface from the
original seed while adding a ``fit`` method and optional GPU support.
"""

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

class FullyConnectedLayer(nn.Module):
    """Multi‑layer perceptron that replaces the toy linear layer.

    Parameters
    ----------
    n_features : int
        Dimensionality of each input sample.
    n_hidden : int, optional
        Number of hidden units in the single hidden layer.
    n_outputs : int, optional
        Dimensionality of the output.
    """
    def __init__(self, n_features: int = 1,
                 n_hidden: int = 16,
                 n_outputs: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 200,
            lr: float = 1e-3,
            batch_size: int = 32,
            device: str = "cpu",
            verbose: bool = False) -> None:
        """Train the network using mean‑squared error."""
        self.to(device)
        dataset = TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = self.forward(xb).squeeze()
                loss = criterion(preds, yb.squeeze())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(loader.dataset)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs}  Loss: {epoch_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions on the given inputs."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(X.astype(np.float32))
            out = self.forward(x).cpu().numpy()
        return out.squeeze()

    def run(self, X: np.ndarray) -> np.ndarray:
        """Compatibility wrapper: ``run`` behaves like ``predict``."""
        return self.predict(X)

def FCL(n_features: int = 1,
        n_hidden: int = 16,
        n_outputs: int = 1) -> FullyConnectedLayer:
    """Factory that mirrors the original API."""
    return FullyConnectedLayer(n_features, n_hidden, n_outputs)

__all__ = ["FCL", "FullyConnectedLayer"]
