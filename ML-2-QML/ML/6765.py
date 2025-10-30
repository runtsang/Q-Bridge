import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data where each sample corresponds to a
    multi‑dimensional feature vector and a target value derived from a smooth
    non‑linear function.  The function mirrors the quantum target used in the
    quantum reference but operates on classical features.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    A lightweight PyTorch dataset that can return either classical feature
    vectors or quantum state tensors.  The ``mode`` parameter controls the
    representation returned in ``__getitem__``.
    """
    def __init__(self, samples: int, num_features: int, mode: str = "classical"):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        self.mode = mode

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        if self.mode == "classical":
            return {
                "features": torch.tensor(self.features[index], dtype=torch.float32),
                "target": torch.tensor(self.labels[index], dtype=torch.float32),
            }
        else:  # quantum mode
            state = torch.tensor(self.features[index], dtype=torch.float32)
            state = F.normalize(state, p=2, dim=0)
            return {
                "states": state,
                "target": torch.tensor(self.labels[index], dtype=torch.float32),
            }

class FullyConnectedLayer(nn.Module):
    """
    A tiny neural module that mimics the behaviour of the quantum fully
    connected layer example.  It applies a linear map followed by a tanh
    activation and returns the mean expectation value.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        expectation = torch.tanh(self.linear(thetas)).mean(dim=0)
        return expectation

class HybridRegressionModel(nn.Module):
    """
    Classical regression model that combines a standard feed‑forward network
    with an optional fully connected layer.  The network can be used as a
    drop‑in replacement for the quantum model while preserving the API
    (``forward(state_batch)`` returns a 1‑D tensor).
    """
    def __init__(self, num_features: int, hidden: int = 32, use_fcl: bool = False):
        super().__init__()
        self.use_fcl = use_fcl
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
        )
        self.fcl = FullyConnectedLayer(hidden) if use_fcl else None
        self.head = nn.Linear(hidden, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        x = self.encoder(state_batch)
        if self.use_fcl:
            x = self.fcl(x)
        return self.head(x).squeeze(-1)

# expose original names for compatibility
QModel = HybridRegressionModel
RegressionDataset = RegressionDataset
generate_superposition_data = generate_superposition_data

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data", "FullyConnectedLayer"]
