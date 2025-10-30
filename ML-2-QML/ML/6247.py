"""Hybrid estimator combining neural‑network and optional sampler evaluation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastHybridEstimator:
    """Evaluate classical neural networks and optional sampler networks."""

    def __init__(self, model: nn.Module, sampler: nn.Module | None = None) -> None:
        self.model = model
        self.sampler = sampler

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute deterministic outputs from the neural network and optionally
        from the sampler.  Each observable receives the concatenated output
        of the network and the sampler’s one‑hot sample."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        if self.sampler is not None:
            self.sampler.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                if self.sampler is not None:
                    # Sample from the probability distribution produced by the sampler
                    probs = self.sampler(inputs)
                    samples = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    onehot = torch.nn.functional.one_hot(samples, num_classes=probs.shape[-1]).float()
                    outputs = torch.cat([outputs, onehot], dim=-1)
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


class FastHybridEstimatorWithNoise(FastHybridEstimator):
    """Adds Gaussian shot‑noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
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


__all__ = ["FastHybridEstimator", "FastHybridEstimatorWithNoise"]
