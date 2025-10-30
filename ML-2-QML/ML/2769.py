"""Classical kernel and estimator utilities."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List

class KernalAnsatz(nn.Module):
    """RBF kernel ansatz with support for scalar or vector gamma."""
    def __init__(self, gamma: float | Sequence[float] = 1.0) -> None:
        super().__init__()
        self.gamma = torch.tensor(gamma, dtype=torch.float32)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist2 = torch.sum(diff * diff, dim=-1)
        return torch.exp(-self.gamma * dist2)

class Kernel(nn.Module):
    """Wraps KernalAnsatz and provides convenient API."""
    def __init__(self, gamma: float | Sequence[float] = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)

    def matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self(x, y).item() for y in b] for x in a])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float | Sequence[float] = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return kernel.matrix(a, b)

class FastBaseEstimator:
    """Evaluate a PyTorch model over parameter sets."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(self, observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row = []
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
    """Adds Gaussian shot noise to the estimator."""
    def evaluate(self, observables, parameter_sets, *, shots: int | None = None, seed: int | None = None):
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = []
        for row in raw:
            noisy.append([float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row])
        return noisy

class QuantumKernelMethod:
    """High-level API combining classical kernel and fast estimator."""
    def __init__(self, gamma: float | Sequence[float] = 1.0, model: nn.Module | None = None):
        self.kernel = Kernel(gamma)
        self.estimator = FastEstimator(model) if model else None

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return self.kernel.matrix(a, b)

    def evaluate(self, observables, parameter_sets, *, shots: int | None = None, seed: int | None = None):
        if self.estimator is None:
            raise ValueError("No model supplied for evaluation.")
        return self.estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "FastBaseEstimator", "FastEstimator", "QuantumKernelMethod"]
