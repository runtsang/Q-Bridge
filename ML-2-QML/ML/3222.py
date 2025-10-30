"""Hybrid classical regression model combining CNN feature extraction and fully connected head.

The dataset generator mirrors the quantum example but the model uses a
convolutional front‑end followed by a fully‑connected projection,
inspired by the Quantum‑NAT CNN architecture.  This design allows
direct comparison with the quantum counterpart while keeping the
classical training pipeline simple.

"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate regression targets from a superposition of angles.

    The data generation logic is identical to the quantum seed: the
    target is a non‑linear function of the sum of input angles.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset for the hybrid regression task."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridRegressionModel(nn.Module):
    """Classical CNN + FC regression model."""
    def __init__(self, num_features: int):
        super().__init__()
        # Reshape 1‑D features into a pseudo‑image for the CNN
        self.input_dim = num_features
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # Compute flattened size after pooling
        dummy = torch.zeros(1, 1, num_features)
        dummy_out = self.cnn(dummy)
        flat_size = dummy_out.view(1, -1).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat_size, 64), nn.ReLU(), nn.Linear(64, 1))
        self.norm = nn.BatchNorm1d(1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # state_batch shape: (bsz, num_features)
        x = state_batch.unsqueeze(1)  # (bsz, 1, num_features)
        features = self.cnn(x)
        flattened = features.view(features.shape[0], -1)
        out = self.fc(flattened)
        return self.norm(out).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
