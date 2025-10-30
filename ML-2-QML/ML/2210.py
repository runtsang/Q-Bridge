import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate float32 features and regression targets."""
    rng = np.random.default_rng()
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that yields float32 states and targets for regression."""
    def __init__(self, samples: int, num_features: int):
        self.x, self.y = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.x[idx], dtype=torch.float32),
            "target": torch.tensor(self.y[idx], dtype=torch.float32),
        }

class QCNNRegressionHybrid(nn.Module):
    """Classical QCNN-inspired regression model."""
    def __init__(self, num_features: int = 8):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(num_features, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return self.head(x).squeeze(-1)

__all__ = ["QCNNRegressionHybrid", "RegressionDataset", "generate_superposition_data"]
