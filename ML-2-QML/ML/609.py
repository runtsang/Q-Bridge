"""Classical regression dataset and model with advanced training utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

def generate_superposition_data(num_features: int, samples: int, noise_std: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data with optional Gaussian noise."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise_std > 0.0:
        y += np.random.normal(0.0, noise_std, size=y.shape).astype(np.float32)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int, noise_std: float = 0.0):
        self.features, self.labels = generate_superposition_data(num_features, samples, noise_std)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class RegressionModel(nn.Module):
    """
    Feed‑forward regression network with optional residual skip.
    Provides a lightweight training API (fit, evaluate, predict) that supports early stopping
    and learning‑rate reduction on plateau.
    """
    def __init__(self, num_features: int, hidden_sizes: list[int] | tuple[int,...] = (32, 16), use_residual: bool = False):
        super().__init__()
        layers = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.use_residual = use_residual
        if use_residual and num_features == 1:
            self.skip = nn.Linear(num_features, 1)
        else:
            self.skip = None

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.net(state_batch.to(torch.float32)).squeeze(-1)
        if self.use_residual and self.skip is not None:
            out += self.skip(state_batch)
        return out

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 10,
        device: str | torch.device = "cpu",
    ) -> None:
        """Train with MSE loss, early stopping, and LR scheduler."""
        self.to(device)
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience//2, verbose=False)
        criterion = nn.MSELoss()
        best_val = float("inf")
        best_state = None
        for epoch in range(epochs):
            self.train()
            for batch in train_loader:
                optimizer.zero_grad()
                pred = self(batch["states"].to(device))
                loss = criterion(pred, batch["target"].to(device))
                loss.backward()
                optimizer.step()
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, device)
                scheduler.step(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                elif epoch - (best_val - val_loss) > patience:
                    break
        if best_state is not None:
            self.load_state_dict(best_state)

    def evaluate(self, loader: DataLoader, device: str | torch.device) -> float:
        self.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                pred = self(batch["states"].to(device))
                total += ((pred - batch["target"].to(device)) ** 2).sum().item()
                count += batch["target"].numel()
        return total / count

    def predict(self, X: torch.Tensor, device: str | torch.device = "cpu") -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self(X.to(device)).cpu()

__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
