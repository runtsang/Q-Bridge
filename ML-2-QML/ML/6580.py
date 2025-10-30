"""Hybrid estimator that combines fast evaluation and optional noise for neural networks.

This module builds on the original FastBaseEstimator by adding support for
Gaussian shot noise and a convolutional preprocessing layer.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class ConvFilter(nn.Module):
    """Convolutional filter that can be prepended to a neural network."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(1)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3])


class HybridEstimator:
    """Hybrid estimator that accepts a torch.nn.Module and optional conv filter.

    The estimator evaluates the model on batches of parameters and returns
    scalar observables.  Optionally, Gaussian shot noise can be added to
    mimic quantum measurement statistics.
    """
    def __init__(self, model: nn.Module, conv: Optional[nn.Module] = None) -> None:
        self.model = model
        self.conv = conv
        self.model.eval()
        if self.conv is not None:
            self.conv.eval()

    def _ensure_batch(self, values: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] = (),
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        if observables is None:
            observables = [lambda outputs: outputs.mean(dim=-1)]
        observables = list(observables)
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                inputs = self._ensure_batch(params)
                if self.conv is not None:
                    inputs = self.conv(inputs)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    scalar = float(value.mean().cpu()) if isinstance(value, torch.Tensor) else float(value)
                    row.append(scalar)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["HybridEstimator", "ConvFilter"]
