"""
RegressionModel: Classical regression with configurable MLP and training helpers.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def generate_superposition_data(num_features: int, samples: int, *,
                                noise_level: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate toy regression data, optionally adding Gaussian noise."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise_level > 0.0:
        y += np.random.normal(scale=noise_level, size=y.shape).astype(np.float32)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapper for the toy regression problem."""
    def __init__(self, samples: int, num_features: int, *, noise_level: float = 0.0):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, noise_level=noise_level
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class RegressionModel(nn.Module):
    """Fully connected network with user‑defined hidden layers and a training helper."""
    def __init__(self, num_features: int, hidden_dims: tuple[int,...] = (32, 16)):
        super().__init__()
        layers = []
        in_dim = num_features
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch).squeeze(-1)

    def fit(
        self,
        train_loader: DataLoader,
        *,
        epochs: int = 100,
        lr: float = 1e-3,
        val_loader: DataLoader | None = None,
        patience: int = 10,
        device: torch.device | str | None = None,
    ) -> None:
        """Train with Adam and early stopping."""
        device = torch.device(device or "cpu")
        self.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        best_val = float("inf")
        bad_epochs = 0

        for epoch in range(epochs):
            self.train()
            for batch in train_loader:
                optimizer.zero_grad()
                preds = self(batch["states"].to(device))
                loss = criterion(preds, batch["target"].to(device))
                loss.backward()
                optimizer.step()

            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        preds = self(batch["states"].to(device))
                        val_loss += criterion(preds, batch["target"].to(device)).item()
                val_loss /= len(val_loader)
                if val_loss < best_val:
                    best_val = val_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                if bad_epochs >= patience:
                    break

    def predict(self, state_batch: torch.Tensor, *, device: torch.device | str | None = None):
        """Convenience prediction wrapper."""
        device = torch.device(device or "cpu")
        self.eval()
        with torch.no_grad():
            return self(state_batch.to(device)).cpu()

__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
