import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data mimicking a quantum superposition distribution.
    The returned labels are two‑dimensional: mean and log‑variance of a Gaussian
    likelihood.  The data generation process uses random angles
    ``theta`` and ``phi`` analogous to the original seed.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    mean = np.sin(angles) + 0.1 * np.cos(2 * angles)
    logvar = 0.5 + 0.1 * np.sin(angles)
    labels = np.stack([mean, logvar], axis=1).astype(np.float32)
    return x, labels

class RegressionDataset(Dataset):
    """
    Dataset yielding state vectors and a two‑dimensional target
    (mean, log‑variance) for probabilistic regression.
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

class ProbabilisticModel(nn.Module):
    """
    Two‑head neural network predicting mean and log‑variance.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(32, 1)
        self.logvar_head = nn.Linear(32, 1)

    def forward(self, state_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(state_batch.to(torch.float32))
        mean = self.mean_head(h).squeeze(-1)
        logvar = self.logvar_head(h).squeeze(-1)
        return mean, logvar

__all__ = ["ProbabilisticModel", "RegressionDataset", "generate_superposition_data"]
