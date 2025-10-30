import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_random_grid(num_samples: int, grid_size: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Create random 4×4 image‑like features and a sinusoidal target."""
    X = np.random.uniform(-1.0, 1.0, size=(num_samples, grid_size * grid_size)).astype(np.float32)
    theta = X.sum(axis=1)
    y = np.sin(theta) + 0.05 * np.cos(2 * theta)
    return X, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, num_samples: int, grid_size: int = 4):
        self.grid_size = grid_size
        self.features, self.labels = generate_random_grid(num_samples, grid_size)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):
        img = self.features[idx].reshape(1, self.grid_size, self.grid_size)
        return {"image": torch.tensor(img, dtype=torch.float32),
                "target": torch.tensor(self.labels[idx], dtype=torch.float32)}

class QuantumRegressionModel(nn.Module):
    """Hybrid CNN‑FC regression model inspired by Quantum‑NAT and the original regression seed."""
    def __init__(self, grid_size: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * (grid_size // 4) * (grid_size // 4), 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.norm = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        out = self.fc(flat).squeeze(-1)
        return self.norm(out)

__all__ = ["RegressionDataset", "QuantumRegressionModel", "generate_random_grid"]
