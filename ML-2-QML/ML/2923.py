"""Unified estimator that can evaluate a PyTorch model for batches of parameters and observables.

The estimator supports optional Gaussian shot noise, mirroring the behaviour of the original FastEstimator.
The API is intentionally identical to the QML counterpart so that the two modules can be used interchangeably
in a hybrid workflow."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D batch tensor for a parameter vector."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class UnifiedEstimator:
    """Evaluate a PyTorch model for batches of parameters and observables.

    The estimator supports optional Gaussian shot noise, mirroring the behaviour of the original FastEstimator.
    It can be used as a drop‑in replacement for FastBaseEstimator in any classical pipeline.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def _evaluate(self,
                  observables: Iterable[ScalarObservable],
                  parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results

    def evaluate(self,
                 observables: Iterable[ScalarObservable],
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 shots: int | None = None,
                 seed: int | None = None) -> List[List[float]]:
        """Compute observables, optionally adding Gaussian shot noise."""
        raw = self._evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["UnifiedEstimator"]
