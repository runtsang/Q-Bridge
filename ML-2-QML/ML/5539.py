"""
QCNNHybrid: Classical model that consumes a quantum‑kernel feature map
and applies a stack of convolution‑like fully connected layers.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# --------------------------------------------------------------------------- #
# 1. Classical RBF kernel (from QuantumKernelMethod.py)
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    """RBF kernel implemented as a PyTorch module."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper exposing a 2‑argument forward signature."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# 2. Fast estimator (from FastBaseEstimator.py)
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Batch evaluator for a PyTorch model with optional Gaussian shot noise."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot‑noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

# --------------------------------------------------------------------------- #
# 3. Helper utilities
# --------------------------------------------------------------------------- #
def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    t = torch.as_tensor(values, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t

# --------------------------------------------------------------------------- #
# 4. Hybrid QCNN model
# --------------------------------------------------------------------------- #
class QCNNHybridModel(nn.Module):
    """
    Classical network that starts from a quantum‑kernel embedding
    and proceeds through a stack of fully‑connected layers
    mirroring the QCNN architecture.
    """
    def __init__(
        self,
        reference_set: Sequence[torch.Tensor],
        kernel_gamma: float = 1.0,
        conv_features: Sequence[int] = (16, 12, 8, 4, 4),
    ) -> None:
        super().__init__()
        self.kernel = Kernel(kernel_gamma)
        self.reference_set = torch.stack(reference_set)  # shape (N, d)
        # Build a linear stack that mimics the QCNN conv/pool pattern
        layers: List[nn.Module] = []
        in_feats = len(reference_set[0])
        for out_feats in conv_features:
            layers.append(nn.Linear(in_feats, out_feats))
            layers.append(nn.Tanh())
            in_feats = out_feats
        layers.append(nn.Linear(in_feats, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel between *x* and the reference set,
        then feed the resulting Gram vector into the conv‑like network.
        """
        # x: (batch, d)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        # Kernel matrix: shape (batch, N)
        gram = torch.stack([self.kernel(x_i, self.reference_set) for x_i in x])
        return torch.sigmoid(self.network(gram))

def QCNNHybrid(reference_set: Sequence[torch.Tensor], kernel_gamma: float = 1.0) -> QCNNHybridModel:
    """Factory that builds a QCNN‑style hybrid model."""
    return QCNNHybridModel(reference_set, kernel_gamma)

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "FastBaseEstimator",
    "FastEstimator",
    "QCNNHybridModel",
    "QCNNHybrid",
]
