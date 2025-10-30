"""Enhanced classical regression estimator with preprocessing, dropout, and training utilities."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Callable

class EstimatorQNN(nn.Module):
    """Feed‑forward regression network with optional regularisation and training helpers."""
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Tuple[int,...] = (8, 4),
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        self.scaler: Optional[StandardScaler] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
        early_stop: Optional[int] = None,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.MSELoss(),
    ) -> None:
        self.scaler = StandardScaler()
        X_np = X.cpu().numpy()
        X_scaled = self.scaler.fit_transform(X_np)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        dataset = TensorDataset(X_tensor, y.to(device))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)
        best_loss = float("inf")
        epochs_no_improve = 0
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self.forward(xb)
                loss = loss_fn(preds.squeeze(), yb.squeeze())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(loader.dataset)
            if early_stop is not None:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= early_stop:
                    break

    def predict(self, X: torch.Tensor, device: str | torch.device = "cpu") -> torch.Tensor:
        if self.scaler is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_np = X.cpu().numpy()
        X_scaled = self.scaler.transform(X_np)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        self.eval()
        with torch.no_grad():
            preds = self.forward(X_tensor).cpu()
        return preds.squeeze()

__all__ = ["EstimatorQNN"]
