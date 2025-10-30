import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Sequence

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data using sinusoidal target."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset yielding classical feature vectors and regression targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class KernalAnsatz(nn.Module):
    """Radial basis function kernel ansatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper exposing a single scalar kernel value."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix between two collections of vectors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class HybridRegressionModel(nn.Module):
    """Hybrid classical‑kernel regression that augments a neural net with RBF kernel features."""
    def __init__(self,
                 num_features: int,
                 kernel_gamma: float = 1.0,
                 ref_size: int = 8):
        super().__init__()
        self.classical_net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.kernel = Kernel(kernel_gamma)
        self.register_buffer("ref_vectors",
                             torch.randn(ref_size, num_features))
        self.final = nn.Linear(16 + ref_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical feature extraction
        h = self.classical_net(x)
        # Quantum kernel features with a fixed reference set
        batch = x.shape[0]
        ref_vectors = self.ref_vectors
        kernel_features = torch.empty(batch, ref_vectors.shape[0], device=x.device)
        for i in range(batch):
            for j in range(ref_vectors.shape[0]):
                kernel_features[i, j] = self.kernel(x[i], ref_vectors[j])
        combined = torch.cat([h, kernel_features], dim=1)
        return self.final(combined).squeeze(-1)

__all__ = [
    "HybridRegressionModel",
    "RegressionDataset",
    "generate_superposition_data",
    "Kernel",
    "kernel_matrix",
]
