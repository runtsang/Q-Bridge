"""Hybrid regression module with a shared encoder and improved training utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def generate_superposition_data(num_features: int, samples: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int, seed: int | None = None):
        self.features, self.labels = generate_superposition_data(num_features, samples, seed)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumRegression__gen324(nn.Module):
    """Classical regression model with batch‑norm, dropout and a learning‑rate scheduler."""

    def __init__(self, num_features: int = 10, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64]
        layers = []
        in_dim = num_features
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.2))
            in_dim = dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        patience: int = 10,
    ) -> "QuantumRegression__gen324":
        """Simple training loop with AdamW and early stopping."""
        device = next(self.parameters()).device
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience // 2, factor=0.5, verbose=True)
        criterion = nn.MSELoss()
        best_val_loss = float("inf")
        patience_counter = 0
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for batch in train_loader:
                states = batch["states"].to(device)
                target = batch["target"].to(device)
                optimizer.zero_grad()
                pred = self.forward(states)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * states.size(0)
            epoch_loss /= len(train_loader.dataset)

            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        states = batch["states"].to(device)
                        target = batch["target"].to(device)
                        pred = self.forward(states)
                        val_loss += criterion(pred, target).item() * states.size(0)
                val_loss /= len(val_loader.dataset)
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        return self

__all__ = ["QuantumRegression__gen324", "RegressionDataset", "generate_superposition_data"]
