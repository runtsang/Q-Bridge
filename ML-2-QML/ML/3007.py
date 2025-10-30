"""Hybrid classical-quantum fully connected layer with estimator utilities.

The class combines a simple linear layer with the evaluation utilities
from the FastBaseEstimator seed. It provides a ``run`` method that
computes the mean tanh activation and an ``evaluate`` method that
applies arbitrary observables to batches of parameter sets.  Optional
Gaussian shot noise can be added to mimic quantum measurement noise.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Ensure 2‑D tensor for batched parameters."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridFCL(nn.Module):
    """Classical hybrid fully‑connected layer with estimator support."""

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Return the mean tanh output for each parameter."""
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute observables for batches of parameter sets.

        Parameters
        ----------
        observables
            Callables that map a tensor of outputs to a scalar or tensor.
        parameter_sets
            Sequence of parameter vectors to evaluate.
        shots
            If provided, Gaussian noise with variance 1/shots is added
            to each observable to emulate measurement shot noise.
        seed
            Random seed for reproducible noise.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).unsqueeze(-1)
                outputs = self(inputs)
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
