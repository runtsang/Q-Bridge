import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_classical_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset that mimics the quantum superposition used in the reference.
    Each sample is a vector of shape (num_features,) and the target is computed
    from the sum of its elements.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_classical_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridQuantumRegression(nn.Module):
    def __init__(self, num_features: int, dropout_prob: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch).squeeze(-1)

    def predict_with_uncertainty(self, state_batch: torch.Tensor, n_samples: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        self.train()
        preds = torch.stack([self(state_batch) for _ in range(n_samples)], dim=0)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        self.eval()
        return mean, std

__all__ = ["HybridQuantumRegression", "RegressionDataset", "generate_classical_superposition_data"]
