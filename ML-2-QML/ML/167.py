"""Extended FastBaseEstimator for PyTorch models with batched evaluation, caching, and optional shot noise."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables.
    Supports GPU, batched inference, caching, and optional Gaussian shot noise.
    """

    def __init__(self, model: nn.Module, device: torch.device | str | None = None) -> None:
        self.model = model.to(device or torch.device("cpu"))
        self.device = torch.device(device or "cpu")
        self._cache = {}
        self.model.eval()

    def _evaluate_batch(
        self, params_batch: torch.Tensor, observables: Iterable[ScalarObservable]
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(params_batch.to(self.device))
        batch_results = []
        for obs in observables:
            val = obs(outputs)
            if isinstance(val, torch.Tensor):
                val = val.mean(-1)
            batch_results.append(val.cpu())
        return torch.stack(batch_results, dim=-1)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        batch_size: int | None = None,
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        if shots is not None and shots < 1:
            raise ValueError("shots must be a positive integer")
        rng = np.random.default_rng(seed)
        results: List[List[float]] = []
        # caching
        cache_key = tuple(tuple(p) for p in parameter_sets)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if shots is None:
                return cached
            else:
                noisy = []
                for row in cached:
                    noisy_row = [
                        float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
                    ]
                    noisy.append(noisy_row)
                return noisy
        # batch processing
        if batch_size is None or batch_size <= 0:
            batch_size = len(parameter_sets)
        for i in range(0, len(parameter_sets), batch_size):
            batch = _ensure_batch(parameter_sets[i : i + batch_size])
            batch_res = self._evaluate_batch(batch, observables)
            results.extend(batch_res.tolist())
        # cache if no shots
        if shots is None:
            self._cache[cache_key] = results
        else:
            noisy = []
            for row in results:
                noisy_row = [
                    float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
                ]
                noisy.append(noisy_row)
            results = noisy
        return results


__all__ = ["FastBaseEstimator"]
