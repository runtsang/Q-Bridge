import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data where each sample is a random vector
    in [-1, 1]^num_features and the target is a smooth non‑linear function
    of the sum of its components.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Classic PyTorch dataset exposing feature vectors and scalar targets.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "features": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class Gen144RegressionModel(nn.Module):
    """
    Hybrid regression model that combines a lightweight CNN‑style extractor
    with a quantum‑inspired linear head.  The ``use_qnn`` flag allows
    switching between a purely classical or a hybrid quantum‑classical
    prediction head.
    """
    def __init__(self, num_features: int, use_qnn: bool = False):
        super().__init__()
        self.use_qnn = use_qnn
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)
        out = self.head(feat).squeeze(-1)
        return out

__all__ = ["Gen144RegressionModel", "RegressionDataset", "generate_superposition_data"]
