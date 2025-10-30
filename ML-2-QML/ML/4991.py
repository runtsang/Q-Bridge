"""Hybrid ConvGen260 class with classical convolution implementation and estimator utilities.

The class emulates a quantum convolution filter but is purely classical using PyTorch.
It also exposes lightweight estimator wrappers that mirror the quantum side,
providing a consistent API for batch evaluation and optional shot noise.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Iterable[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor

class FastBaseEstimator:
    """Evaluate a PyTorch model for multiple parameter sets."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Iterable[Iterable[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
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
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Same as FastBaseEstimator but adds Gaussian shot noise."""
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Iterable[Iterable[float]],
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

class ConvGen260(nn.Module):
    """Classic convolution filter that mimics the quantum `Conv` interface."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

    def __call__(self, data) -> float:
        return self.run(data)

__all__ = ["ConvGen260", "FastBaseEstimator", "FastEstimator"]
