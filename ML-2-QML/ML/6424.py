"""Enhanced FastBaseEstimator with batched GPU inference and optional shot noise."""
from __future__ import annotations

import math
from typing import Callable, Iterable, List, Sequence

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence of floats into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Fastest estimator for PyTorch models with optional Gaussian shot noise.

    The estimator accepts a :class:`torch.nn.Module` and evaluates a list of
    observables on a batch of parameter sets.  Optional shot noise can be
    injected by specifying ``shots``; a random seed is accepted for reproducible
    experiments.  The implementation automatically uses the GPU if available
    and supports batched inference for maximum throughput.
    """

    def __init__(self, model: nn.Module, device: torch.device | str | None = None) -> None:
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate the model for every parameter set and observable.

        Parameters
        ----------
        observables:
            Iterable of callables that map a model output tensor to a scalar.
            If empty, the mean of the last dimension is used.
        parameter_sets:
            Sequence of parameter vectors.  Each vector is fed to the model as a
            single input example.
        shots:
            If provided, Gaussian noise with standard deviation ``1 / sqrt(shots)``
            is added to each output value to mimic measurement shot noise.
        seed:
            Random seed for reproducible noise generation.
        """
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        raw: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            batch = _ensure_batch(parameter_sets).to(self.device)
            outputs = self.model(batch)
            for row_idx in range(outputs.shape[0]):
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs[row_idx])
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                raw.append(row)

        if shots is None:
            return raw

        rng = np.random.default_rng(seed)
        sigma = 1.0 / math.sqrt(shots)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [rng.normal(loc=mean, scale=max(sigma, 1e-6)) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def confidence_intervals(
        self,
        results: List[List[float]],
        shots: int,
        confidence: float = 0.95,
    ) -> List[List[tuple[float, float]]]:
        """Return confidence intervals for noisy results.

        Parameters
        ----------
        results:
            Raw or noisy results as returned by :meth:`evaluate`.
        shots:
            Number of shots used to generate the noise.
        confidence:
            Confidence level (default 0.95).
        """
        z = {
            0.80: 1.282,
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
        }[confidence]
        sigma = 1.0 / math.sqrt(shots)
        intervals: List[List[tuple[float, float]]] = []
        for row in results:
            row_ints = [(m - z * sigma, m + z * sigma) for m in row]
            intervals.append(row_ints)
        return intervals


__all__ = ["FastBaseEstimator"]
