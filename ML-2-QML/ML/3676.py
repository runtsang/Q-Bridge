"""Hybrid kernel implementation using classical RBF and shot‑noise estimation.

This module extends the original classical RBF kernel with a lightweight
estimator that can inject Gaussian shot noise, mirroring the behaviour
of the quantum fast estimator while remaining fully classical.
"""

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Callable, Sequence

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results


class HybridKernel(nn.Module):
    """Classical RBF kernel augmented with a noisy estimator.

    The class can compute a kernel matrix and evaluate a PyTorch model
    with optional Gaussian shot noise, providing a convenient interface
    for benchmarking against its quantum counterpart.
    """

    def __init__(self, gamma: float = 1.0, shots: int | None = None, seed: int | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.shots = shots
        self.seed = seed

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

    def evaluate_with_noise(
        self,
        model: nn.Module,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate a model and optionally add Gaussian shot noise."""
        estimator = FastBaseEstimator(model)
        raw = estimator.evaluate(observables, parameter_sets)
        if self.shots is None:
            return raw
        rng = np.random.default_rng(self.seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy
