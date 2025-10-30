import torch
from torch import nn
import numpy as np

class QCNNHybrid(nn.Module):
    """
    Classical convolution–style network with an optional quantum layer.
    The quantum layer must be a callable that accepts a tensor and returns a tensor.
    """
    def __init__(self,
                 num_features: int,
                 use_quantum: bool = False,
                 qnn: nn.Module | None = None):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(num_features, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

        self.use_quantum = use_quantum
        self.qnn = qnn

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        if self.use_quantum and self.qnn is not None:
            x = self.qnn(x)
        return torch.sigmoid(self.head(x))

def QCNNHybridFactory(num_features: int,
                      use_quantum: bool = False,
                      qnn: nn.Module | None = None) -> QCNNHybrid:
    """
    Factory that returns a QCNNHybrid instance.
    """
    return QCNNHybrid(num_features, use_quantum, qnn)

# ------------------------------
# Regression data helpers (adapted from QuantumRegression seed)
# ------------------------------

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate toy regression data based on a superposition of |0> and |1>."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

__all__ = ["QCNNHybrid", "QCNNHybridFactory",
           "RegressionDataset", "generate_superposition_data"]
