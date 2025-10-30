"""Hybrid classical model that fuses a convolutional filter, an RBF kernel embedding, and a linear head.

The module also includes synthetic regression utilities inspired by the quantum regression example.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence

# ----------------------------------------------------------------------
# Classical RBF kernel utilities
# ----------------------------------------------------------------------
class RBFAnsatz(nn.Module):
    """Simple RBF kernel ansatz."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class RBFKernel(nn.Module):
    """Wrapper around :class:`RBFAnsatz`."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.ansatz = RBFAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix between two sets of vectors."""
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ----------------------------------------------------------------------
# Hybrid model
# ----------------------------------------------------------------------
class QuanvolutionHybrid(nn.Module):
    """
    Classical hybrid model:
    * 2×2 convolutional filter to extract image patches.
    * RBF kernel embedding of the flattened feature map against a set of prototypes.
    * Linear head for classification or regression.
    """
    def __init__(self, num_classes: int = 10, gamma: float = 1.0):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.kernel = RBFKernel(gamma)
        self.prototypes = None  # to be set via ``set_prototypes``.
        self.head = nn.Linear(0, num_classes)  # placeholder; will be re‑initialised after prototypes.

    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        """
        Register a fixed set of prototype vectors against which the RBF kernel is evaluated.
        Parameters
        ----------
        prototypes : torch.Tensor
            Shape (P, D) where P is the number of prototypes and D is the feature dimension.
        """
        self.prototypes = prototypes
        # Re‑initialise the head to match the number of prototypes.
        self.head = nn.Linear(prototypes.shape[0], self.head.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.prototypes is None:
            raise RuntimeError("Prototypes not set. Call ``set_prototypes`` before forward.")
        # Extract patches and flatten.
        feats = self.conv(x)  # (B, 4, H, W)
        feats = feats.view(feats.size(0), -1)  # (B, D)
        # Compute kernel similarity to each prototype.
        B, D = feats.shape
        P = self.prototypes.shape[0]
        # Expand for broadcasting.
        feats_exp = feats.unsqueeze(1).expand(B, P, D)
        proto_exp = self.prototypes.unsqueeze(0).expand(B, P, D)
        # Kernel matrix per batch.
        kernel_vals = torch.exp(-self.kernel.ansatz.gamma * torch.sum((feats_exp - proto_exp) ** 2, dim=-1))
        # Linear head.
        logits = self.head(kernel_vals)
        return F.log_softmax(logits, dim=-1)

# ----------------------------------------------------------------------
# Synthetic regression utilities (borrowed from the quantum example)
# ----------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data based on superposition states."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapping the synthetic regression data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

__all__ = [
    "QuanvolutionHybrid",
    "RBFAnsatz",
    "RBFKernel",
    "kernel_matrix",
    "generate_superposition_data",
    "RegressionDataset",
]
