import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data.
    Features are uniformly sampled in [-1, 1].
    Labels are a non‑linear combination of the sum of features.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset wrapping the synthetic data for training.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridMLModel(nn.Module):
    """
    Classical regression network that can optionally concatenate
    quantum‑encoded features.
    """
    def __init__(self, num_features: int, quantum_feature_dim: int = 0):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.quantum_dim = quantum_feature_dim
        if quantum_feature_dim:
            self.final = nn.Linear(16 + quantum_feature_dim, 1)
        else:
            self.final = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor, q_features: torch.Tensor | None = None):
        h = self.main(x)
        if self.quantum_dim and q_features is not None:
            h = torch.cat([h, q_features], dim=-1)
        return self.final(h).squeeze(-1)

__all__ = ["generate_superposition_data", "RegressionDataset", "HybridMLModel"]
