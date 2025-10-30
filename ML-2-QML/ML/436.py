"""Enhanced regression neural network with regularization and training utilities."""

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

class SharedClassName(nn.Module):
    """Feed‑forward regressor with configurable depth, dropout, and batch norm.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list[int] | None, optional
        Sizes of hidden layers. Defaults to [8, 4].
    dropout : float, optional
        Dropout probability after each hidden layer.
    use_batch_norm : bool, optional
        Whether to add batch norm after each hidden layer.
    """
    def __init__(self, input_dim: int = 2,
                 hidden_dims: list[int] | None = None,
                 dropout: float = 0.0,
                 use_batch_norm: bool = False):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [8, 4]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def fit(self,
            X: torch.Tensor,
            y: torch.Tensor,
            batch_size: int = 32,
            epochs: int = 200,
            lr: float = 1e-3,
            val_split: float = 0.1,
            verbose: bool = False):
        """Train the network using MSE loss and Adam optimizer."""
        dataset = TensorDataset(X, y)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        optimizer = Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            self.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = self(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                val_loss = self.evaluate(val_loader, criterion)
                print(f"Epoch {epoch+1}/{epochs} - Val MSE: {val_loss:.4f}")

    def evaluate(self, loader: DataLoader, criterion: nn.Module) -> float:
        self.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in loader:
                pred = self(xb)
                losses.append(criterion(pred, yb).item())
        return sum(losses) / len(losses)

def EstimatorQNN() -> SharedClassName:
    """Compatibility wrapper returning the default SharedClassName."""
    return SharedClassName()

__all__ = ["EstimatorQNN", "SharedClassName"]
