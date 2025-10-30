"""Combined classical regression model incorporating kernel and MLP."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate amplitude‑encoded samples and regression targets."""
    theta = np.random.uniform(-np.pi, np.pi, size=samples)
    phi = np.random.uniform(-np.pi, np.pi, size=samples)
    # Build real‑valued feature vector (cosθ, sinθ·cosφ, sinθ·sinφ) for simplicity
    x = np.vstack([np.cos(theta), np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)]).T
    y = np.sin(2 * theta) * np.cos(phi)
    return x.astype(np.float32), y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset producing amplitude‑encoded samples and regression targets."""
    def __init__(self, samples: int, num_features: int):
        self.states, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class KernalAnsatz(nn.Module):
    """Radial basis function kernel ansatz."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wraps :class:`KernalAnsatz` to provide a kernel matrix."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

class QuantumRegressionGen270(nn.Module):
    """Hybrid MLP that optionally concatenates RBF kernel features."""
    def __init__(self, num_features: int, use_kernel: bool = False, gamma: float = 1.0):
        super().__init__()
        self.use_kernel = use_kernel
        input_dim = num_features + (1 if use_kernel else 0)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        if self.use_kernel:
            self.kernel = Kernel(gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_kernel:
            # Compute self‑kernel feature (scalar per sample) and concatenate
            k = self.kernel(x, x).unsqueeze(-1)
            features = torch.cat([x, k], dim=-1)
        else:
            features = x
        return self.mlp(features).squeeze(-1)

__all__ = ["generate_superposition_data", "RegressionDataset", "Kernel", "QuantumRegressionGen270"]
