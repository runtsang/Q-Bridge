import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a mixed‑state regression dataset.

    Returns:
        features: random points in [-1, 1]^{num_features}
        labels: sinusoidal target with added cosine noise
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapping the superposition data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridFunction(nn.Module):
    """Differentiable sigmoid used as a hybrid head."""

    def __init__(self, shift: float = 0.0):
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x + self.shift)

class ClassicalRegressionHybrid(nn.Module):
    """Hybrid regression model combining a classical dense head with a quantum feature extractor."""

    def __init__(self,
                 num_features: int,
                 quantum_layer: nn.Module | None = None,
                 shift: float = 0.0):
        super().__init__()
        self.num_features = num_features
        self.quantum_layer = quantum_layer

        # Classical dense branch
        self.dense = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_features),
        )
        # Final prediction head
        self.head = HybridFunction(shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical feature extraction
        classical_feat = self.dense(x)

        # Quantum feature extraction (if provided)
        if self.quantum_layer is not None:
            # quantum_layer expects a batch of state vectors
            q_feat = self.quantum_layer(x)
            # Combine by element‑wise addition
            feat = classical_feat + q_feat
        else:
            feat = classical_feat

        # Prediction
        return self.head(feat).squeeze(-1)

__all__ = ["generate_superposition_data",
           "RegressionDataset",
           "HybridFunction",
           "ClassicalRegressionHybrid"]
