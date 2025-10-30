import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic features and sinusoidal targets."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Classic regression dataset mirroring the quantum counterpart."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class SamplerModule(nn.Module):
    """Classical soft‑max sampler with a bottleneck and dropout regulariser."""
    def __init__(self, in_features: int = 2, hidden_size: int = 4, out_features: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_size, out_features),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

class QModel(nn.Module):
    """Feed‑forward regression head used in the hybrid model."""
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch).squeeze(-1)

class UnifiedSamplerRegressor(nn.Module):
    """
    Dual‑output model that jointly learns a probability distribution
    and a continuous target from the same input.
    """
    def __init__(self, num_features: int = 2, hidden_size: int = 4):
        super().__init__()
        self.sampler = SamplerModule(in_features=num_features, hidden_size=hidden_size)
        self.regressor = QModel(num_features)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        probs = self.sampler(inputs)
        target = self.regressor(inputs)
        return {"probs": probs, "target": target}

__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "SamplerModule",
    "QModel",
    "UnifiedSamplerRegressor",
]
