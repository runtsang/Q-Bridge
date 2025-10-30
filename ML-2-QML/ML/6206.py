import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Sequence

# --------------------------------------------------------------------------- #
# Classical RBF kernel with a learnable bandwidth
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    """Classical radial‑basis function kernel with trainable gamma."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper that normalises inputs and returns a scalar kernel value."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix between two collections of feature vectors."""
    k = Kernel(gamma)
    return np.array([[k(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Dataset and regression utilities
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for regression."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that samples superposition states and a synthetic target."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Simple feed‑forward network for regression on feature vectors
# --------------------------------------------------------------------------- #
class ClassicalMLP(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# --------------------------------------------------------------------------- #
# Hybrid kernel‑regressor
# --------------------------------------------------------------------------- #
class UnifiedKernelRegressor(nn.Module):
    """
    Hybrid regression model that can operate in three modes:
    * ``classical`` – uses only the RBF kernel.
    * ``quantum``   – uses only the quantum kernel.
    * ``hybrid``    – concatenates both kernel feature maps.
    The model is fully differentiable; when ``hybrid`` the quantum part
    is treated as a fixed feature extractor (no gradients w.r.t. quantum
    parameters), but the final linear head is trainable.
    """
    def __init__(self,
                 mode: str = "hybrid",
                 num_features: int = 4,
                 gamma: float = 1.0,
                 quantum_module: str = None):
        super().__init__()
        assert mode in {"classical", "quantum", "hybrid"}
        self.mode = mode
        self.num_features = num_features
        self.gamma = gamma

        # Classical kernel
        self.classical_kernel = Kernel(gamma) if mode in {"classical", "hybrid"} else None

        # Quantum kernel (lazy import to keep ML side free of torchquantum)
        if mode in {"quantum", "hybrid"}:
            if quantum_module is None:
                raise ValueError("Quantum module name must be supplied for quantum modes.")
            mod = __import__(quantum_module)
            self.q_kernel = mod.Kernel()
        else:
            self.q_kernel = None

        # Feature dimension
        feat_dim = num_features
        if mode == "hybrid":
            feat_dim = num_features * 2
        self.head = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of samples, shape (B, D).
        Returns
        -------
        torch.Tensor
            Predicted target, shape (B,).
        """
        features = []
        if self.mode in {"classical", "hybrid"}:
            # Classical RBF feature: use kernel of each sample with itself
            classical_feat = self.classical_kernel(x, x).diag().unsqueeze(-1)
            features.append(classical_feat)
        if self.mode in {"quantum", "hybrid"}:
            # Quantum kernel feature: evaluate kernel between each sample
            # and itself (diagonal of Gram matrix)
            q_feat = self.q_kernel(x, x).diag().unsqueeze(-1)
            features.append(q_feat)
        feat = torch.cat(features, dim=-1)
        return self.head(feat).squeeze(-1)

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "RegressionDataset",
    "generate_superposition_data",
    "ClassicalMLP",
    "UnifiedKernelRegressor",
]
