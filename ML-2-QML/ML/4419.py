"""Classical regression model incorporating convolution‑like layers and a hybrid head.

The module defines a dataset, a QCNN‑style encoder, and a simple linear head.
It can be trained with standard PyTorch tools and serves as the classical
counterpart to the quantum implementation below."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic superposition data for regression."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset yielding feature vectors and target values."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QCNNModel(nn.Module):
    """Convolution‑inspired encoder mirroring the QCNN helper."""
    def __init__(self, in_features: int = 8, out_features: int = 4) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(in_features, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class HybridRegression(nn.Module):
    """Classical regression network with a convolution‑like encoder and a linear head."""
    def __init__(self, num_features: int, conv_out: int = 4) -> None:
        super().__init__()
        self.encoder = QCNNModel(in_features=num_features, out_features=conv_out)
        self.head = nn.Linear(conv_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.head(x).squeeze(-1)

__all__ = ["RegressionDataset", "generate_superposition_data", "HybridRegression"]
