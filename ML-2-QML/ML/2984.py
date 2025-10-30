"""Hybrid estimator that supports classical PyTorch models with optional shot noise."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastHybridEstimator:
    """Evaluate neural networks for batches of inputs and observables with optional shot noise.

    This class extends the original FastBaseEstimator by adding support for Gaussian shot noise
    and a convenient interface for regression tasks. It is fully compatible with the
    original API and can be used as a drop‑in replacement.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Compute the expectation of each observable for every parameter set.

        Parameters
        ----------
        observables:
            Callables that map a model output tensor to a scalar.
            If empty, the mean of the last dimension is used.
        parameter_sets:
            Iterable of parameter vectors (floats) that will be fed to the model.
        shots, seed:
            Optional Gaussian noise parameters that emulate finite‑shot sampling.
        """
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

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


# Backward‑compatibility aliases
FastBaseEstimator = FastHybridEstimator
FastEstimator = FastHybridEstimator


__all__ = ["FastHybridEstimator", "FastBaseEstimator", "FastEstimator"]
