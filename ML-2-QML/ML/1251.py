"""EstimatorQNNGen: classical feed‑forward regressor with training utilities."""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

class EstimatorQNNGen(nn.Module):
    """
    A fully‑connected regression network that supports configurable hidden layers,
    optional dropout, and out‑of‑the‑box training/evaluation helpers.
    """
    def __init__(self, input_dim: int = 2, hidden_sizes: list[int] | tuple[int,...] = (8, 4), dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def train_model(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module = nn.MSELoss(),
        epochs: int = 20,
        device: str = "cpu",
    ) -> None:
        """
        Simple training loop that prints epoch‑wise loss.
        """
        self.to(device)
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                pred = self(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    def evaluate(
        self,
        loader: DataLoader,
        loss_fn: nn.Module = nn.MSELoss(),
        device: str = "cpu",
    ) -> float:
        """
        Return the mean‑squared error over the provided loader.
        """
        self.to(device)
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = self(batch_x)
                loss = loss_fn(pred, batch_y)
                total_loss += loss.item() * batch_x.size(0)
        return total_loss / len(loader.dataset)

    @classmethod
    def from_config(cls, config: dict):
        """
        Instantiate from a configuration dictionary.
        """
        return cls(
            input_dim=config.get("input_dim", 2),
            hidden_sizes=config.get("hidden_sizes", (8, 4)),
            dropout=config.get("dropout", 0.0),
        )

__all__ = ["EstimatorQNNGen"]
