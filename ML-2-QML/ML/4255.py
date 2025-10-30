import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data based on superposition states."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapping the synthetic superposition data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ConvFilter(nn.Module):
    """Classical 2‑D convolution filter emulating the quantum quanvolution block."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Reshape to a small 2‑D patch for the filter
        patch = data.view(-1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(patch)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3], keepdim=True)  # scalar per sample

class HybridRegressionModel(nn.Module):
    """Hybrid classical‑quantum regression model.

    The input is first processed by ConvFilter to extract a low‑dimensional
    representation.  The resulting scalar is fed into a small MLP that
    mimics the fully‑connected segment of the Quantum‑NAT architecture.
    """
    def __init__(self, num_features: int, hidden_dims: list[int] = [32, 16]) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=2)
        layers = []
        prev_dim = 1
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(state_batch)
        return self.mlp(conv_out).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
