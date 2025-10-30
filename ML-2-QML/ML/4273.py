import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class KernalAnsatz(nn.Module):
    """RBF kernel transformation used as a feature expansion."""
    def __init__(self, gamma: float = 1.0, n_basis: int = 10):
        super().__init__()
        self.gamma = gamma
        self.n_basis = n_basis
        self.register_buffer("refs", torch.randn(n_basis, 1))

    def forward(self, x: torch.Tensor):
        # x: [batch, features]
        diff = x.unsqueeze(1) - self.refs  # [batch, n_basis, features]
        return torch.exp(-self.gamma * torch.sum(diff**2, dim=2)).squeeze(-1)  # [batch, n_basis]

class QCNNModel(nn.Module):
    """Classical approximation of the quantum convolution network."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(hidden_dim // 4, hidden_dim // 8), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(hidden_dim // 8, hidden_dim // 8), nn.Tanh())
        self.head = nn.Linear(hidden_dim // 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x)).squeeze(-1)

class RegressionDataset(Dataset):
    """Dataset that returns kernel‑expanded features together with the label."""
    def __init__(self, samples: int, num_features: int, gamma: float = 1.0):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        self.kernel = KernalAnsatz(gamma=gamma, n_basis=10)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        raw = torch.tensor(self.features[idx], dtype=torch.float32)
        kernel_feat = self.kernel(raw.unsqueeze(0))  # [1, n_basis]
        return {"states": torch.cat([raw, kernel_feat.squeeze(0)]), "target": torch.tensor(self.labels[idx], dtype=torch.float32)}

class HybridRegressionModel(nn.Module):
    """Classical hybrid model that combines kernel expansion and QCNN feature extraction."""
    def __init__(self, input_dim: int, kernel_dim: int = 10):
        super().__init__()
        self.qcnn = QCNNModel(input_dim + kernel_dim)
        self.head = nn.Linear(8, 1)  # final linear layer after QCNN

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qcnn(x)
        return self.head(x).squeeze(-1)

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
