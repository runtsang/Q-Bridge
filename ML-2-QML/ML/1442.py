import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int, noise_std: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Superposition‑style regression data with optional Gaussian noise."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = np.sum(x, axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    y += np.random.normal(0, noise_std, size=y.shape)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Classic regression dataset with optional noise."""
    def __init__(self, samples: int, num_features: int):
        self.x, self.y = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"states": torch.tensor(self.x[idx], dtype=torch.float32),
                "target": torch.tensor(self.y[idx], dtype=torch.float32)}

class ResidualBlock(nn.Module):
    """A single residual block with batch‑norm and ReLU."""
    def __init__(self, dim: int):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(x))
        out = self.fc1(out)
        out = self.relu(self.bn2(out))
        out = self.fc2(out)
        return self.relu(out + residual)

class QModel(nn.Module):
    """Deep residual neural network for regression."""
    def __init__(self, num_features: int, hidden_dim: int = 64, n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_layer = nn.Linear(num_features, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(state_batch)
        x = self.blocks(x)
        x = self.dropout(x)
        return self.output_layer(x).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
