"""Hybrid estimator for classical neural networks with optional shot noise.

The class wraps a :class:`torch.nn.Module` and evaluates it over batches of
parameter sets.  Observables are user‑supplied callables that map the network
output tensor to a scalar.  An optional ``shots`` argument adds Gaussian
shot‑noise to mimic measurement uncertainty."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a neural network on parameter sets with optional shot noise."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """Return a list of observable values for each parameter set.

        Parameters
        ----------
        observables:
            Callables that map the network output tensor to a scalar.
        parameter_sets:
            Iterable of sequences of floats that are fed as a single‑row
            batch to the network.
        shots:
            If provided, Gaussian noise with variance ``1/shots`` is added to
            each observable value to mimic measurement shot noise.
        seed:
            Random seed for reproducibility of the noise.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                batch = _ensure_batch(params)
                outputs = self.model(batch)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
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


__all__ = ["FastBaseEstimator"]
