"""Hybrid kernel method combining classical RBF and optional fully connected feature transform.

This module provides a lightweight, pure‑Python implementation that can be used
in place of the original ``QuantumKernelMethod`` class.  The class exposes
methods to compute a Gram matrix either with a standard radial basis
function or with a learned linear feature map.  It also ships a small
sampler network and a regression toy model that mirror the quantum
counterparts in the seed repository.

The design intentionally keeps the API identical to the original seed
so that downstream code can switch between the classical and quantum
implementations without modification.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------

class HybridKernelMethod(nn.Module):
    """Classical RBF kernel with optional linear feature map.

    Parameters
    ----------
    gamma : float
        Width of the RBF kernel.
    n_features : int
        Dimensionality of the linear feature map.  If set to ``None`` the
        kernel operates on the raw input.
    use_fcl : bool
        When ``True`` a 1‑layer fully connected network is applied before
        the kernel.  This mirrors the quantum ``FullyConnectedLayer``.
    """
    def __init__(self, gamma: float = 1.0, n_features: int | None = None,
                 use_fcl: bool = False) -> None:
        super().__init__()
        self.gamma = gamma
        self.use_fcl = use_fcl
        if use_fcl:
            # a tiny linear layer followed by tanh to emulate the quantum
            # fully‑connected layer used in the reference.
            self.fcl = nn.Linear(1, 1)
            self.n_features = n_features or 1
        else:
            self.fcl = None
            self.n_features = n_features

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Optional feature map."""
        if self.fcl is not None:
            # broadcast to match expected shape
            x = x.unsqueeze(-1) if x.ndim == 1 else x
            return torch.tanh(self.fcl(x))
        return x

    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel for two 1‑D tensors."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value for a pair of samples."""
        x = self._preprocess(x)
        y = self._preprocess(y)
        return self.rbf_kernel(x, y)

    def kernel_matrix(self, a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor]) -> np.ndarray:
        """Construct the Gram matrix for two collections of samples."""
        return np.array([[float(self(x, y)) for y in b] for x in a])

# ---------------------------------------------------------------------------

def SamplerQNN() -> nn.Module:
    """Return a tiny neural network that mimics the quantum sampler.

    The network maps a 2‑dimensional input to a probability distribution
    over two outcomes.  It is intentionally very small so that it can
    be used for quick sanity checks.
    """
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return F.softmax(self.net(inputs), dim=-1)

    return SamplerModule()

# ---------------------------------------------------------------------------

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for the regression toy problem."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that returns raw feature vectors and a scalar target."""
    def __init__(self, samples: int, num_features: int) -> None:
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """Simple feed‑forward regression network."""
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch.to(torch.float32)).squeeze(-1)

__all__ = [
    "HybridKernelMethod",
    "SamplerQNN",
    "RegressionDataset",
    "generate_superposition_data",
    "QModel",
]
