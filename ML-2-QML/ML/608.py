"""Enhanced feed‑forward regressor with training utilities."""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


class EstimatorQNN(nn.Module):
    """A deep fully‑connected regression model with regularisation and a train helper."""

    def __init__(self, input_dim: int = 2, hidden_dims: tuple[int,...] = (64, 32, 16), dropout: float = 0.2) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hdim),
                    nn.BatchNorm1d(hdim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

    @staticmethod
    def train(
        model: "EstimatorQNN",
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
    ) -> tuple["EstimatorQNN", list[float]]:
        """Train the model on the provided data and return the trained model and loss history."""
        model = model.to(device)
        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=lr)

        loss_hist: list[float] = []
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
            loss_hist.append(loss.item())
        return model, loss_hist


__all__ = ["EstimatorQNN"]
