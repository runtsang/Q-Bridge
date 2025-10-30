"""Hybrid classical fully connected layer with batch evaluation and optional shot noise."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

import numpy as np
import torch
from torch import nn


ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats to a 2‑D torch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridFCL(nn.Module):
    """A classical fully‑connected layer that supports batch evaluation and observables."""

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Compute tanh‑activated linear output."""
        return torch.tanh(self.linear(params))

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Compute scalar observables for a batch of parameter sets.

        Parameters
        ----------
        observables : Iterable[Callable]
            Callables that map the network output to a scalar.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors.
        shots : int, optional
            If provided, Gaussian noise with variance 1/shots is added to each mean.
        seed : int, optional
            Seed for the random number generator used when adding shot noise.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                batch = _ensure_batch(params)
                outputs = self(batch)
                row: List[float] = []
                for observable in observables:
                    val = observable(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy

        return results


__all__ = ["HybridFCL"]
