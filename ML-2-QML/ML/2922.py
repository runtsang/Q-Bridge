import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int):
    """Generate synthetic data by sampling angles and computing a sinusoidal target."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapping the synthetic superposition data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QModel(nn.Module):
    """Simple feed‑forward regression model used as a classical baseline."""
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class QCNNGen188(nn.Module):
    """
    Classical analogue of a quantum convolutional neural network.
    It mimics the layer ordering of the QCNN ansatz using fully connected
    blocks and simple pooling operations implemented as linear transformations.
    """
    def __init__(self, input_dim: int = 8, conv_dims: list[int] | None = None):
        super().__init__()
        if conv_dims is None:
            conv_dims = [16, 16, 12, 8, 4, 4]
        layers = []
        prev_dim = input_dim
        for out_dim in conv_dims:
            layers.append(nn.Linear(prev_dim, out_dim))
            layers.append(nn.Tanh())
            prev_dim = out_dim
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return torch.sigmoid(self.head(x))

__all__ = ["QCNNGen188", "RegressionDataset", "QModel", "generate_superposition_data"]
