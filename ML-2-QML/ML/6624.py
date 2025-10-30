import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Tuple

class EstimatorQNN(nn.Module):
    """
    A versatile fully‑connected neural network for regression.
    Supports arbitrary hidden depth, dropout, batch‑norm, and activation functions.
    Designed to be plug‑in for standard PyTorch training loops.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Tuple[int,...] = (64, 32),
        output_dim: int = 1,
        dropout: float = 0.0,
        batch_norm: bool = False,
        activation: nn.Module = nn.Tanh(),
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def train_loop(
        model: "EstimatorQNN",
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int = 100,
        early_stop_patience: int = 10,
        device: str = "cpu",
    ) -> "EstimatorQNN":
        """
        Minimal training loop with early stopping on validation loss.
        """
        model = model.to(device)
        best_val = float("inf")
        patience = 0
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    val_losses.append(criterion(preds, yb).item())
            val_loss = sum(val_losses) / len(val_losses)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), "best_estimatorqnn.pt")
                patience = 0
            else:
                patience += 1
                if patience >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        model.load_state_dict(torch.load("best_estimatorqnn.pt"))
        return model

    @staticmethod
    def evaluate(
        model: "EstimatorQNN",
        data_loader: DataLoader,
        device: str = "cpu",
    ) -> dict:
        """
        Compute MSE and MAE over the provided data_loader.
        """
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in data_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds.append(model(xb).cpu())
                targets.append(yb.cpu())
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        return {
            "mse": mean_squared_error(targets.numpy(), preds.numpy()),
            "mae": mean_absolute_error(targets.numpy(), preds.numpy()),
        }

__all__ = ["EstimatorQNN"]
