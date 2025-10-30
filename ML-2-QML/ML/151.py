"""Classical regression with advanced training utilities."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic superposition data with added Gaussian noise."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    noise = 0.05 * np.random.randn(samples).astype(np.float32)
    return x, (y + noise).astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset wrapper for the synthetic regression data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class RegressionModel(nn.Module):
    """Feed‑forward network with dropout and batch normalization for regression."""
    def __init__(self, num_features: int, hidden_sizes: tuple[int,...] = (64, 32), dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 200,
        lr: float = 1e-3,
        device: str = "cpu",
        patience: int | None = 20,
    ) -> None:
        """Train the network with Adam and optional early stopping."""
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = F.mse_loss
        best_val = float("inf")
        no_improve = 0

        for epoch in range(1, epochs + 1):
            self.train()
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                preds = self(batch["states"].to(device))
                loss = criterion(preds, batch["target"].to(device))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch["states"].size(0)
            epoch_loss /= len(train_loader.dataset)

            if val_loader is not None:
                val_loss = self.evaluate(val_loader, device=device)
                if val_loss < best_val:
                    best_val = val_loss
                    no_improve = 0
                    torch.save(self.state_dict(), "_best_model.pt")
                else:
                    no_improve += 1
                if patience is not None and no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | Train loss: {epoch_loss:.4f}" + (f" | Val loss: {val_loss:.4f}" if val_loader else ""))

        if val_loader is not None:
            self.load_state_dict(torch.load("_best_model.pt"))

    def predict(self, loader: DataLoader, device: str = "cpu") -> torch.Tensor:
        """Return predictions for a dataset."""
        self.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                preds.append(self(batch["states"].to(device)).cpu())
        return torch.cat(preds)

    def evaluate(self, loader: DataLoader, device: str = "cpu") -> float:
        """Compute mean squared error on a dataset."""
        preds = self.predict(loader, device=device)
        targets = torch.cat([batch["target"] for batch in loader])
        return F.mse_loss(preds, targets).item()


__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
