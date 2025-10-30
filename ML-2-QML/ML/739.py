"""Enhanced classical regression model with data augmentation and training utilities."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def generate_augmented_data(num_features: int, samples: int, noise_std: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data with optional Gaussian noise."""
    X = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    y += np.random.normal(0.0, noise_std, size=y.shape).astype(np.float32)
    return X, y

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int, noise_std: float = 0.05):
        self.features, self.labels = generate_augmented_data(num_features, samples, noise_std)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

class QModel(nn.Module):
    """MLP with optional dropout for regression."""
    def __init__(self, num_features: int, hidden_sizes: list[int] | None = None, dropout: float = 0.0):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        layers = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def fit(self, train_loader: DataLoader, epochs: int = 20, lr: float = 1e-3, device: str | torch.device = "cpu") -> None:
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = self(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}")

    def predict(self, X: torch.Tensor, device: str | torch.device = "cpu") -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self(X.to(device)).cpu()

__all__ = ["QModel", "RegressionDataset", "generate_augmented_data"]
