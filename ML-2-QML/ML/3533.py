"""Hybrid classical kernel and regression utilities.

This module extends the original classical RBF kernel implementation
with quantum‑kernel support, enabling a hybrid kernel that blends
classical and quantum similarity measures.  It also provides a
straightforward neural‑network regression model and a convenient
wrapper for training on hybrid features.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Sequence, Optional

# ------------------------------------------------------------------
# Classical kernel utilities
# ------------------------------------------------------------------
class KernalAnsatz(nn.Module):
    """Radial basis function (RBF) kernel ansatz (compatibility layer)."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper that exposes the RBF kernel as a torch module."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix for sequences ``a`` and ``b`` using the RBF kernel."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ------------------------------------------------------------------
# Quantum kernel utilities (placeholder for hybrid usage)
# ------------------------------------------------------------------
class QuantumKernelMatrix:
    """Container for a pre‑computed quantum kernel matrix.

    In a real workflow the matrix would be supplied by a quantum backend
    and can be combined with the classical RBF kernel using
    :class:`HybridKernel`.
    """
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

# ------------------------------------------------------------------
# Hybrid kernel combining classical and quantum similarity
# ------------------------------------------------------------------
class HybridKernel:
    """Hybrid kernel that adds a classical RBF kernel to a quantum kernel.

    Parameters
    ----------
    gamma : float
        RBF kernel bandwidth.
    quantum_matrix : np.ndarray | None
        Pre‑computed quantum kernel matrix.  If omitted, only the
        classical kernel is used.
    """

    def __init__(self, gamma: float = 1.0, quantum_matrix: Optional[np.ndarray] = None) -> None:
        self.gamma = gamma
        self.quantum_matrix = quantum_matrix

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return the hybrid Gram matrix."""
        # Classical part
        x = torch.tensor(a, dtype=torch.float32)
        y = torch.tensor(b, dtype=torch.float32)
        classical = kernel_matrix(x, y, self.gamma)
        # Quantum part
        if self.quantum_matrix is not None:
            return classical + self.quantum_matrix
        return classical

# ------------------------------------------------------------------
# Dataset generation (classical superposition data)
# ------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data that mimics a quantum superposition."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Torch dataset wrapping the synthetic data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# ------------------------------------------------------------------
# Classical regression model
# ------------------------------------------------------------------
class QModel(nn.Module):
    """Simple feed‑forward regression network."""
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)

# ------------------------------------------------------------------
# Hybrid regression wrapper
# ------------------------------------------------------------------
class HybridRegression:
    """Convenience wrapper that trains a classical network
    using a hybrid kernel‑based feature map.

    The network is trained on the classical RBF features; the
    quantum kernel can be fed as an additional feature vector.
    """
    def __init__(self, num_features: int, gamma: float = 1.0, quantum_matrix: Optional[np.ndarray] = None):
        self.gamma = gamma
        self.model = QModel(num_features)
        self.quantum_matrix = quantum_matrix

    def _classical_features(self, X_t: torch.Tensor) -> torch.Tensor:
        """Pairwise RBF kernel matrix (diagonal only)."""
        dists = torch.cdist(X_t, X_t, p=2)
        return torch.exp(-self.gamma * dists**2)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 1e-3):
        """Train the network on the hybrid features."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        for _ in range(epochs):
            optimizer.zero_grad()
            feats = self._classical_features(X_t).diag().unsqueeze(-1)
            if self.quantum_matrix is not None:
                feats = torch.cat([feats, torch.tensor(self.quantum_matrix.diagonal(), dtype=torch.float32).unsqueeze(-1)], dim=-1)
            pred = self.model(feats)
            loss = criterion(pred, y_t)
            loss.backward()
            optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions using hybrid features."""
        X_t = torch.tensor(X, dtype=torch.float32)
        feats = self._classical_features(X_t).diag().unsqueeze(-1)
        if self.quantum_matrix is not None:
            feats = torch.cat([feats, torch.tensor(self.quantum_matrix.diagonal(), dtype=torch.float32).unsqueeze(-1)], dim=-1)
        with torch.no_grad():
            return self.model(feats).cpu().numpy()

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "QuantumKernelMatrix",
    "HybridKernel",
    "generate_superposition_data",
    "RegressionDataset",
    "QModel",
    "HybridRegression",
]
