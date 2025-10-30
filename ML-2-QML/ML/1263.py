"""ML implementation of EstimatorQNNGen209.

This module defines a flexible feed‑forward regressor that extends the original
tiny network with dropout, batch‑normalisation and an optional early‑stopping
mechanism.  The public API mirrors the seed: a function EstimatorQNN() that
returns an instance of the class.
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

class EstimatorQNNGen209(nn.Module):
    """
    A lightweight yet extensible feed‑forward regressor.

    Parameters
    ----------
    input_dim : int, default 2
        Number of input features.
    hidden_dims : Sequence[int], default (8, 4)
        Sizes of successive hidden layers.
    dropout : float, default 0.1
        Drop‑out probability applied after each hidden layer.
    batchnorm : bool, default True
        Whether to insert a BatchNorm1d layer after each hidden layer.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: tuple[int,...] = (8, 4),
        dropout: float = 0.1,
        batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the regression output."""
        return self.net(x)

    # ------------------------------------------------------------------
    # Convenience training helpers
    # ------------------------------------------------------------------
    def train_one_epoch(
        self,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device | str = "cpu",
    ) -> float:
        """Train for one epoch and return the mean loss."""
        self.train()
        total_loss = 0.0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = self(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        return total_loss / len(dataloader.dataset)

    def evaluate(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device | str = "cpu",
    ) -> float:
        """Return the mean loss on the validation set."""
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in dataloader:
                xb, yb = xb.to(device), yb.to(device)
                pred = self(xb)
                loss = criterion(pred, yb)
                total_loss += loss.item() * xb.size(0)
        return total_loss / len(dataloader.dataset)

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 32,
        lr: float = 1e-3,
        epochs: int = 200,
        patience: int | None = 10,
        device: torch.device | str = "cpu",
    ) -> list[float]:
        """Fit the network to data and return training loss history."""
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        self.to(device)

        history = []
        best_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(epochs):
            loss = self.train_one_epoch(loader, optimizer, criterion, device)
            history.append(loss)
            if loss < best_loss:
                best_loss = loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if patience is not None and epochs_no_improve >= patience:
                break

        self.to("cpu")
        return history

    def predict(self, X: torch.Tensor, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return predictions for the given inputs."""
        self.eval()
        X = X.to(device)
        with torch.no_grad():
            return self(X).cpu()

def EstimatorQNN() -> EstimatorQNNGen209:
    """Return an instance of the upgraded EstimatorQNNGen209."""
    return EstimatorQNNGen209()
