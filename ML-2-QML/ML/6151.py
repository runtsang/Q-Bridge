import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset with polynomial and interaction terms.
    The target is a non‑linear function of the augmented features.
    """
    # Base random features
    x = np.random.randn(samples, num_features).astype(np.float32)
    # Add polynomial and interaction features
    poly_features = np.concatenate(
        [x, x ** 2, x[:, :1] * x[:, 1:2] if num_features > 1 else np.empty((samples, 0), dtype=np.float32)],
        axis=1
    )
    angles = poly_features.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return poly_features, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch dataset wrapping the synthetic regression data.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ResidualBlock(nn.Module):
    """
    Simple residual block with batch‑norm and ReLU.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class QRegressionModel(nn.Module):
    """
    Classical regression model with residual blocks, dropout and a linear head.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_blocks: int = 2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.dropout = nn.Dropout(p=0.3)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(state_batch)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x)
        return self.output_layer(x).squeeze(-1)


__all__ = ["QRegressionModel", "RegressionDataset", "generate_superposition_data"]
