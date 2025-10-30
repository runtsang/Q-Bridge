import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data by sampling angles from a uniform distribution
    and computing a noisy sinusoidal target.  The data is returned as ``float32`` arrays
    suitable for feeding into a PyTorch model.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles) + 0.05 * np.random.randn(samples)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    PyTorch dataset wrapping the synthetic superposition data.  The ``states`` tensor
    contains the raw features and ``target`` holds the regression labels.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        # Standardise features to zero mean, unit variance
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features).astype(np.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """
    A small feed‑forward neural network with batch‑normalisation, dropout and a
    flexible hidden‑layer configuration.  The network is designed to be a drop‑in
    replacement for the original seed model while offering improved generalisation.
    """
    def __init__(self, num_features: int, hidden_dims: tuple[int,...] = (64, 32), dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = num_features
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)

    def predict(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper that returns the model output without the extra
        ``squeeze`` used in the training loop.
        """
        return self.net(state_batch.to(torch.float32))

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
