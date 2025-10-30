"""
ML module for EstimatorQNNGen117.

This module defines a lightweight feed‑forward network together with a
FastEstimator that can evaluate batches of inputs and observables.
The estimator supports optional Gaussian shot noise, making it suitable
for hybrid quantum‑classical experiments.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of scalars into a 2‑D tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a neural network for a collection of inputs.

    Parameters
    ----------
    model
        A :class:`torch.nn.Module` that maps an input tensor to an output.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a list of rows, each row containing the results of all
        observables for a single parameter set.
        """
        if not observables:
            observables = [lambda out: out.mean(dim=-1)]
        self.model.eval()
        results: List[List[float]] = []
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
    """Adds optional Gaussian shot noise to :class:`FastBaseEstimator`."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
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


class EstimatorQNNGen117(nn.Module):
    """A small, fully‑connected regression network.

    The architecture mirrors the original EstimatorQNN but is extended
    to expose a ``evaluate`` method that accepts a list of observable
    callables and an arbitrary number of input rows.
    """

    def __init__(self, hidden_sizes: Sequence[int] = (8, 4)) -> None:
        super().__init__()
        layers = []
        in_dim = 2
        for size in hidden_sizes:
            layers.extend([nn.Linear(in_dim, size), nn.Tanh()])
            in_dim = size
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(inputs)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """Convenience wrapper that delegates to :class:`FastEstimator`."""
        estimator = FastEstimator(self)
        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)


__all__ = ["EstimatorQNNGen117", "FastBaseEstimator", "FastEstimator"]
